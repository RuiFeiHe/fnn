import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
from torch import randint
from torch.multiprocessing import spawn, Queue
from torch.utils.data import Dataset as TorchDataset, DataLoader


# -------------- Loader Base --------------

class Loader:
    """Loader"""

    def __call__(self, training=True, display_progress=True):
        raise NotImplementedError()


# -------------- Torch Dataset Wrapper --------------

class FnnTorchDataset(TorchDataset):
    """将 fnn Dataset 包装成 PyTorch Dataset，供 DataLoader 使用"""

    def __init__(self, fnn_dataset, sample_size, training=True):
        self.fnn_dataset = fnn_dataset
        self.sample_size = sample_size
        self.keys = fnn_dataset.keys(training=training)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        high = self.fnn_dataset.df.loc[key].samples - self.sample_size
        if high > 0:
            index = randint(high=high, size=(1,)).item() + np.arange(self.sample_size)
        else:
            index = np.arange(self.sample_size)
        return self.fnn_dataset.load(key, index)


def collate_fn(batch):
    """将 list of dict 合并成 dict of stacked arrays"""
    result = {}
    for k in batch[0].keys():
        result[k] = np.stack([b[k] for b in batch], axis=1)
    return result


# -------------- Loader Types --------------

class DatasetLoader(Loader):
    """Dataset Loader"""

    def _init(self, dataset):
        raise NotImplementedError()


class Batches(DatasetLoader):
    """Randomly Sampled Batches — 支持 num_workers 并行加载"""

    def __init__(self, sample_size, batch_size, training_size, validation_size,
                 num_workers=4, prefetch_factor=2):
        """
        Parameters
        ----------
        sample_size : int
        batch_size : int
        training_size : int
        validation_size : int
        num_workers : int
            并行加载的 worker 数量（默认 4）
        prefetch_factor : int
            每个 worker 预取的 batch 数（默认 2）
        """
        assert sample_size > 0
        assert batch_size > 0
        assert training_size >= 0
        assert validation_size >= 0

        self.sample_size = int(sample_size)
        self.batch_size = int(batch_size)
        self.training_size = int(training_size)
        self.validation_size = int(validation_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)

    def _init(self, dataset):
        assert dataset.df.samples.min() >= self.sample_size
        self.dataset = dataset

    def __call__(self, training=True, display_progress=True):
        size = self.training_size if training else self.validation_size
        desc = "Training Batches" if training else "Validation Batches"

        if not size:
            return

        torch_dataset = FnnTorchDataset(self.dataset, self.sample_size, training=training)

        if not len(torch_dataset):
            return

        # 随机采样 size * batch_size 个样本（有放回）
        indices = np.random.randint(0, len(torch_dataset), size=size * self.batch_size).tolist()
        subset = torch.utils.data.Subset(torch_dataset, indices)

        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,          # 已经随机采样了 indices
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            pin_memory=True,        # 加速 CPU→GPU 传输
            persistent_workers=False,  # 改成 False，每 epoch 重建 worker 避免泄漏
        )

        iterator = tqdm(loader, desc=desc, total=size) if display_progress else loader

        for batch in iterator:
            yield batch


# -------------- Miscellaneous Loaders --------------

class EmptyLoader(Loader):
    """Empty Loader"""

    def __init__(self, training_size, validation_size):
        assert training_size >= 0
        assert validation_size >= 0
        self.training_size = int(training_size)
        self.validation_size = int(validation_size)

    def __call__(self, training=True, display_progress=True):
        if training:
            iterations = range(self.training_size)
            desc = "Training"
        else:
            iterations = range(self.validation_size)
            desc = "Validation"

        if display_progress:
            iterations = tqdm(iterations, desc=desc)

        for _ in iterations:
            yield dict()