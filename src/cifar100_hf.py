"""get cifar100 without the (very slow) official server.

the images come from the huggingface parquet files and are written back in the exact
layout torchvision expects (root/cifar-100-python/{train,test,meta}), same pixels and labels.
"""
import io
import json
import pickle
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

HF_URL = 'https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100'
SPLITS = ('train', 'test')


def download_parquet(parquet_dir):
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = parquet_dir / f'{split}.parquet'
        if not target.exists():
            urllib.request.urlretrieve(f'{HF_URL}/{split}-00000-of-00001.parquet', target)
    return parquet_dir


def build_cifar100(parquet_dir, root):
    """convert the parquet files to torchvision's pickle files, returns root/cifar-100-python"""
    parquet_dir, base = Path(parquet_dir), Path(root) / 'cifar-100-python'
    base.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        df = pd.read_parquet(parquet_dir / f'{split}.parquet')
        imgs = np.stack([np.array(Image.open(io.BytesIO(d['bytes'])).convert('RGB')) for d in df['img']])
        entry = {
            'data': imgs.transpose(0, 3, 1, 2).reshape(len(df), -1),  # N x 3072, R then G then B planes
            'fine_labels': df['fine_label'].tolist(),
            'coarse_labels': df['coarse_label'].tolist(),
        }
        with open(base / split, 'wb') as f:
            pickle.dump(entry, f)

    # the class names live in the parquet schema
    schema = pq.read_schema(parquet_dir / 'train.parquet')
    names = json.loads(schema.metadata[b'huggingface'])['info']['features']['fine_label']['names']
    with open(base / 'meta', 'wb') as f:
        pickle.dump({'fine_label_names': names}, f)
    return base


def prepare_cifar100(root='data/cifar100', cache_dir=None):
    """make root/cifar-100-python ready. cache_dir (e.g. on google drive) keeps the converted
    files so a fresh colab runtime doesn't have to download and convert again"""
    root = Path(root)
    base = root / 'cifar-100-python'
    if all((base / n).is_file() for n in (*SPLITS, 'meta')):
        return base

    cached = Path(cache_dir) / 'data_cifar100' if cache_dir else None
    if cached and cached.is_dir():
        shutil.copytree(cached, root, dirs_exist_ok=True)
    else:
        build_cifar100(download_parquet(root.parent / 'cifar100_hf'), root)
        if cached:
            shutil.copytree(root, cached, dirs_exist_ok=True)

    assert all((base / n).is_file() for n in (*SPLITS, 'meta')), 'cifar100 missing'
    return base
