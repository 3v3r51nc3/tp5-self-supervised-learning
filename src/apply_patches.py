"""Patch a fresh clone of cs294-158-ssl so it supports:
  - the new `patch_position` task (patch_position.py is copied into deepul_helper/tasks/)
  - CIFAR100 as a second dataset for all the tasks

usage: python apply_patches.py <path to cs294-158-ssl> <path to patch_position.py>
"""
import shutil
import sys
from pathlib import Path

MARKER = '# [tp5 patch]'

# the official cifar server is really slow, so cifar100 is rebuilt from the huggingface parquet files
# (same pixels and labels, but not the same bytes) so torchvision's md5 checks can't pass on it
LOCAL_CIFAR100 = """class LocalCIFAR100(datasets.CIFAR100):
    def _check_integrity(self):
        names = [name for name, _ in self.train_list + self.test_list]
        return all(osp.isfile(osp.join(self.root, self.base_folder, n)) for n in names)

    def _load_meta(self):
        import pickle
        with open(osp.join(self.root, self.base_folder, self.meta['filename']), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.classes = data[self.meta['key']]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
"""


def patch_file(path, edits):
    text = path.read_text()
    if MARKER in text:
        print(f'{path.name}: already patched, skipping')
        return
    for old, new, count in edits:
        assert text.count(old) == count, f'{path.name}: expected {count}x {old!r}, found {text.count(old)}'
        text = text.replace(old, new)
    path.write_text(f'{MARKER}\n' + text)
    print(f'{path.name}: patched')


def main(repo, patch_module):
    repo = Path(repo)
    pkg = repo / 'deepul_helper'

    shutil.copy(patch_module, pkg / 'tasks' / 'patch_position.py')
    print('patch_position.py copied')

    patch_file(pkg / 'tasks' / '__init__.py', [
        ('from .simclr import SimCLR', 'from .simclr import SimCLR\nfrom .patch_position import PatchPosition', 1),
    ])

    # cifar100 uses the same code path as cifar10 (same image size)
    patch_file(pkg / 'tasks' / 'rotation.py', [
        ("dataset == 'cifar10'", "dataset.startswith('cifar')", 3),
    ])
    patch_file(pkg / 'tasks' / 'simclr.py', [
        ("dataset == 'cifar10'", "dataset.startswith('cifar')", 1),
    ])

    patch_file(pkg / 'data.py', [
        ("dataset == 'cifar10'", "dataset.startswith('cifar')", 4),
        ("    elif task == 'rotation':",
         """    elif task == 'patch_position':
        # stays 32x32 here, the model upsamples to 64x64 on the gpu (the colab cpu only has 2 cores)
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        if train:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        else:
            transform = transforms.Compose([transforms.ToTensor(), norm])
    elif task == 'rotation':""", 1),
        ("datasets.CIFAR10(osp.join('data', dataset), train=True,",
         "cifar_cls(osp.join('data', dataset), train=True,", 1),
        ("datasets.CIFAR10(osp.join('data', dataset), train=False,",
         "cifar_cls(osp.join('data', dataset), train=False,", 1),
        ("    elif dataset.startswith('cifar'):\n",
         "    elif dataset.startswith('cifar'):\n"
         "        cifar_cls = datasets.CIFAR10 if dataset == 'cifar10' else LocalCIFAR100\n", 1),
        ("def get_datasets(dataset, task):", LOCAL_CIFAR100 + "\n\ndef get_datasets(dataset, task):", 1),
    ])

    patch_file(pkg / 'demos.py', [
        ("    elif task == 'simclr':\n        model = SimCLR(dataset, n_classes, None)\n",
         "    elif task == 'simclr':\n        model = SimCLR(dataset, n_classes, None)\n"
         "    elif task == 'patch_position':\n        model = PatchPosition(dataset, n_classes)\n", 1),
    ])

    patch_file(repo / 'train_self_supervised_task.py', [
        ("from warmup_scheduler import GradualWarmupScheduler",
         "try:\n    from warmup_scheduler import GradualWarmupScheduler\nexcept ImportError:\n"
         "    GradualWarmupScheduler = None  # only needed with -u > 0", 1),
        ("num_workers=16", "num_workers=min(16, os.cpu_count() or 2)", 2),
        ("assert args.task in ['context_encoder', 'rotation', 'cpc', 'simclr']",
         "assert args.task in ['context_encoder', 'rotation', 'cpc', 'simclr', 'patch_position']", 1),
        ("    elif args.task == 'simclr':\n        model = SimCLR(args.dataset, n_classes, dist)\n",
         "    elif args.task == 'simclr':\n        model = SimCLR(args.dataset, n_classes, dist)\n"
         "    elif args.task == 'patch_position':\n        model = PatchPosition(args.dataset, n_classes)\n", 1),
    ])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
