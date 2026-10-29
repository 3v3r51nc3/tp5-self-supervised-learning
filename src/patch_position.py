import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchPosition(nn.Module):
    """Pretext task from Doersch et al. 2015 (arXiv 1505.05192).

    Take a query patch (center of a 3x3 grid) and one of its 8 neighbours,
    the network has to guess which of the 8 positions the neighbour came from.
    """
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    # cifar images (32x32) get upsampled to 64x64 on the gpu in encode/forward, then the grid is:
    # 3 patches of 16px + 2 gaps of 6px = 60px, so 2px of margin on each side for the jitter
    IMG_SIZE = 64
    PATCH = 16
    GAP = 6
    JITTER = 2

    # (row, col) offset of the neighbour in grid cells, the label is the index in this list
    NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    def __init__(self, dataset, n_classes, color_drop=True):
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes
        # color dropping is there to stop the net from cheating with chromatic aberration
        self.color_drop = color_drop

        # same conv trunk for the 2 patches (siamese) and for the full image at test time
        self.encoder = nn.Sequential(
            self._block(3, 64, 1), self._block(64, 64, 2),      # 16 -> 8
            self._block(64, 128, 1), self._block(128, 128, 2),  # 8 -> 4
            self._block(128, 256, 1), self._block(256, 256, 2), # 4 -> 2
        )
        # a 16x16 patch ends up as 2x2, a 64x64 image as 8x8 so pool it down to 2x2 as well
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.latent_dim = 256 * 2 * 2

        self.head = nn.Sequential(
            nn.Linear(2 * self.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, len(self.NEIGHBOURS)),
        )

    @staticmethod
    def _block(in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def construct_classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes),
        )

    def _features(self, x):
        return torch.flatten(self.pool(self.encoder(x)), 1)

    def encode(self, images):
        # full images only, patches go through _features directly
        if images.shape[-1] != self.IMG_SIZE:
            images = F.interpolate(images, size=self.IMG_SIZE, mode='bilinear', align_corners=False)
        return self._features(images)

    def forward(self, images):
        batch_size = images.shape[0]
        if images.shape[-1] != self.IMG_SIZE:
            images = F.interpolate(images, size=self.IMG_SIZE, mode='bilinear', align_corners=False)
        query, neighbour, targets = self._make_pairs(images)

        # both patches go through the same encoder, so run them as one batch
        feats = self._features(torch.cat((query, neighbour), dim=0))
        f_query, f_neighbour = feats[:batch_size], feats[batch_size:]
        logits = self.head(torch.cat((f_query, f_neighbour), dim=1))

        loss = F.cross_entropy(logits, targets)
        acc = logits.argmax(dim=-1).eq(targets).float().mean() * 100.

        # the representation used for the linear classifier is the full image (no patches, no color drop)
        # no grad on it since the classifier is trained separately in the training loop
        with torch.no_grad():
            zs = self._features(images)

        return dict(Loss=loss, Acc1=acc), zs

    def _make_pairs(self, images):
        batch_size, device = images.shape[0], images.device
        stride = self.PATCH + self.GAP
        origin = (self.IMG_SIZE - (3 * self.PATCH + 2 * self.GAP)) // 2
        center = origin + stride  # top left corner of the middle cell

        targets = torch.randint(len(self.NEIGHBOURS), (batch_size,), device=device)
        offsets = torch.tensor(self.NEIGHBOURS, device=device)[targets]  # (B, 2)

        # independent jitter for each patch so the gap size can't be used as a clue
        jitter = torch.randint(-self.JITTER, self.JITTER + 1, (2, batch_size, 2), device=device)
        query_pos = center + jitter[0]
        neighbour_pos = center + offsets * stride + jitter[1]

        query = self._crop(images, query_pos)
        neighbour = self._crop(images, neighbour_pos)
        if self.color_drop:
            query, neighbour = self._drop_colors(query), self._drop_colors(neighbour)
        return query, neighbour, targets

    def _crop(self, images, pos):
        # pos is (B, 2) with the (row, col) of the top left corner of each patch
        batch_size, device = images.shape[0], images.device
        steps = torch.arange(self.PATCH, device=device)
        rows = (pos[:, 0:1] + steps)[:, None, :, None]
        cols = (pos[:, 1:2] + steps)[:, None, None, :]
        batch_idx = torch.arange(batch_size, device=device)[:, None, None, None]
        chan_idx = torch.arange(3, device=device)[None, :, None, None]
        return images[batch_idx, chan_idx, rows, cols]

    @staticmethod
    def _drop_colors(patches):
        # keep one random channel and replace the other two by gaussian noise (like in the paper)
        batch_size, device = patches.shape[0], patches.device
        keep = torch.randint(3, (batch_size,), device=device)
        mask = F.one_hot(keep, 3).to(patches.dtype)[:, :, None, None]  # (B, 3, 1, 1)

        kept = (patches * mask).sum(dim=1, keepdim=True)
        std = kept.std(dim=(1, 2, 3), keepdim=True)
        noise = torch.randn_like(patches) * std
        return patches * mask + noise * (1. - mask)
