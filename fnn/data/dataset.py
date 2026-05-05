import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Any
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)
# -------------- Data Item --------------


class Item:
    """Data Item"""

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int | slice | 1D array
            used to index the data item

        Returns
        -------
        ND array
            data array
        """
        raise NotImplementedError()


class NpyFile(Item):
    """Numpy File Data"""

    def __init__(self, data):
        """
        Parameters
        ----------
        data : ND array
            data array
        """
        self.fd = tempfile.mkdtemp()
        self.fp = os.path.join(self.fd, "data.npy")
        np.save(self.fp, data)

    def __getitem__(self, index):
        return np.load(self.fp, mmap_mode="r")[index]

    def __del__(self):
        os.remove(self.fp)
        os.rmdir(self.fd)


# -------------- Data Set --------------


class Dataset:
    """Data Set"""

    def __init__(self, dataframe):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            index -- unique identifier
            columns -- datainfo (`training`, `samples`) and dataitems
        """
        self.df = dataframe.astype(self.datainfo)

        assert np.unique(dataframe.index).size == len(dataframe), "Index is not unique"

        for item in self.dataitems:
            # assert isinstance(dataframe[item][0], Item), f"{dataframe[item][0]} is not an instance of Item"
            # import ipdb
            # ipdb.set_trace(context=20)
            assert isinstance(dataframe[item].iloc[0], Item), f"{dataframe[item][0]} is not an instance of Item"

    @property
    def datainfo(self):
        return {"training": bool, "samples": int}

    @property
    def dataitems(self):
        return set(self.df.columns) - set(self.datainfo)

    def keys(self, training=True):
        """
        Parameters
        ----------
        training : bool
            training or validation

        Returns
        -------
        pandas.Index
            training or validation keys
        """
        if training:
            return self.df.index[self.df.training]
        else:
            return self.df.index[~self.df.training]

    def load(self, key, index=None):
        """
        Parameters
        ----------
        key : hashable
            label for DataFrame.loc
        index : int | slice | 1D array
            used to index the data item

        Returns
        -------
        dict[str, ND array]
            data items
        """
        data = self.df.loc[key]

        if index is None:
            return {item: data[item][:] for item in self.dataitems}
        else:
            return {item: data[item][index] for item in self.dataitems}

# -------------- Training and evaluating new digital twin --------------

def load_training_data(directory, max_items=None):
    """
    Load dataset to train digital twin.
    Parameters
    ----------
    directory : str | Path
        directory containing the dataset files
    Returns
    -------
    Dataset
        loaded dataset
    """
    directory = Path(directory)
    target_cols = ['training', 'samples', 'stimuli', 'perspectives', 'modulations', 'units']
    col_data = {}
    for col_path in sorted(directory.iterdir()):
        col_name = col_path.stem
        if col_name not in target_cols or not col_path.is_dir():
            continue

        all_files = sorted(col_path.iterdir())
        if max_items is not None:
            all_files = all_files[:max_items]
        value_dict = {}
        for file_path in tqdm(all_files, desc=f"Loading {col_name}"):
            if col_name in ['training', 'samples']:
                value_dict[file_path.stem] = np.load(file_path).item()
            else:
                value_dict[file_path.stem] = NpyFile(np.load(file_path))

        col_data[col_name] = value_dict
        
    df = pd.DataFrame(col_data)
    df.index.name = 'trial_id'
    
    if df.isna().any().any():
        missing = {
            col: df.index[df[col].isna()].tolist()
            for col in df.columns if df[col].isna().any()
        }
        raise ValueError(f"Missing data detected: {missing}")

    return Dataset(df[target_cols])


def recursive_load(path: Path, load_fn: Callable[[Path], Any] = None):
    """
    Recursively load files from a directory structure using a custom loader.

    Parameters
    ----------
    path : Path
        Root directory to traverse.
    load_fn : Callable[[Path], Any], optional
        A function that takes a Path and returns loaded data.
        If None, the Path objects are returned as-is.

    Returns
    -------
    list
        A nested list mirroring the directory structure, containing either
        loaded data or Path objects.
    """
    entries = sorted(path.iterdir())
    result = []
    for entry in entries:
        if entry.is_dir():
            result.append(recursive_load(entry, load_fn))
        else:
            result.append(load_fn(entry) if load_fn else entry)
    return result


def load_evaluation_data(directory: Path):
    """
    Load dataset to evaluate digital twin.
    Parameters
    ----------
    directory : str | Path
        directory containing the dataset files
    Returns
    -------
        dictionary with contents of directory indexed by subdirectory name
    """
    contents = {}
    subdirs = []
    for subdir in directory.iterdir():
        if subdir.is_dir():
            subdirs.append(subdir)
    for subdir in tqdm(subdirs, desc="Subdirectories"):
            logger.info(f"Loading {subdir.name}...")
            contents[subdir.name] = recursive_load(subdir, load_fn=np.load)
    return contents
