import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from .. import paths
from ..file import check_path
from ..file.write_utils import write_fasta


class ProteinDataset(Data.Dataset):
    def __init__(self, name, path=None, sequence=None, metadata=None, dir='dataset'):
        self.name = name  # the name of the dataset
        self.path = path if path is not None else os.path.join(paths[dir], name)  # the storage directory of the dataset

        self.csv = os.path.join(self.path, f'metadata.csv')
        self.fasta = os.path.join(self.path, f'sequence.fasta')
        self.metadata = pd.DataFrame(columns=['name', 'sequence', 'length'])  # include index, name, sequence, label, partition, etc.
        self.features = {}  # feature data: {feature_name: feature_data}, feature_data: np.ndarray
        # the default index of self.metadata is range(len(self.metadata)), which is consistent with the default index of self.features

        if sequence is not None:
            assert metadata is None, 'Either sequence or dataframe should be provided, not both'
            seq_name = [f'seq_{i}' for i in range(len(sequence))]
            length = [len(seq) for seq in sequence]
            self.metadata = pd.DataFrame({'name': seq_name, 'sequence': sequence, 'length': length})

        if metadata is not None:
            assert sequence is None, 'Either sequence or dataframe should be provided, not both'
            self.metadata = metadata
            if 'length' not in self.metadata.columns:
                self.metadata['length'] = self.metadata['sequence'].apply(len)

        # load metadata if the csv file exists
        if sequence is None and metadata is None and os.path.exists(self.csv):
            self.load_metadata()

        # assign feature_state and h5_index_map after loading metadata
        self.feature_state = {name: False for name in self.features.keys()}  # whether the feature data is loaded
        self.h5_index_map = {i: i for i in range(len(self.metadata))}  # map the index of the subset to the original dataset for lazy loading

    @property
    def df(self):
        return self.metadata

    @property
    def dataframe(self):
        return self.metadata

    @property
    def sequences(self):
        return self.metadata['sequence'].values

    def add_metadata(self, data, column_name='new_column', **kwargs):
        if column_name in self.metadata.columns:
            print(f'Column [{column_name}] already exists, overwriting the data')

        # if data is not a dict, the default order of data is consistent with that in the dataframe.
        if isinstance(data, list):
            self.metadata[column_name] = data  # not align based on index
        elif isinstance(data, np.ndarray):
            self.metadata[column_name] = data  # not align based on index
        elif isinstance(data, pd.Series):
            self.metadata[column_name] = data.values  # not align based on index
        elif isinstance(data, dict):
            map_key = kwargs.get('map_key', 'index')
            self.metadata[column_name] = self.metadata[map_key].map(data)  # align based on index
        elif isinstance(data, pd.DataFrame):
            left_on = kwargs.get('left_on', 'sequence')
            right_on = kwargs.get('right_on', 'sequence')
            how = kwargs.get('how', 'inner')
            self.metadata = pd.merge(self.metadata, data, left_on=left_on, right_on=right_on, how=how)
        else:
            raise ValueError(f'Unsupported data type {type(data)}')

    def add_feature(self, features: dict):
        for feature_name, feature_data in features.items():
            self.features[feature_name] = feature_data
            self.feature_state[feature_name] = True

    def append(self, new_metadata, new_features=None):
        # missing some columns will be filled with NaN
        assert isinstance(new_metadata, pd.DataFrame), 'new_metadata should be a pandas DataFrame'
        self.metadata = pd.concat([self.metadata, new_metadata], ignore_index=True)
        self.metadata['length'] = self.metadata['sequence'].apply(len)
        if new_features is not None:
            assert all(self.feature_state.values()), 'Feature data should be loaded before appending data'
            for feature_name, feature_data in new_features.items():
                self.features[feature_name] = np.concatenate([self.features[feature_name], feature_data], axis=0)
        # do not save the dataset automatically to avoid frequent I/O operations

    def remove(self, indices):
        # all feature data should be loaded before removing data to avoid misalignment
        assert all(self.feature_state.values()), 'Feature data should be loaded before removing data'
        assert set(indices) <= set(self.metadata.index), 'index out of range'
        self.metadata = self.metadata.drop(indices).reset_index(drop=True)
        for feature_name, feature_data in self.features.items():
            self.features[feature_name] = np.delete(feature_data, indices, axis=0)
        # do not save the dataset automatically to avoid frequent I/O operations

    def save(self, overwrite=True, features=None, fasta_description=False):
        # save metadata
        if not os.path.exists(self.csv) or overwrite:
            check_path(self.csv)
            self.metadata.to_csv(self.csv, index=False)

        # save sequence
        if not os.path.exists(self.fasta) or overwrite:
            check_path(self.fasta)
            index = self.metadata['index'] if 'index' in self.metadata else self.metadata.index.tolist()
            index = [str(h) for h in index]
            if fasta_description:  # write metadata to description
                description = self.metadata.to_dict(orient='records')
                for row in description:
                    row.pop('sequence')
                write_fasta(self.fasta, self.metadata['sequence'], custom_index=index, description=description)
            else:  # only write index to header
                write_fasta(self.fasta, self.metadata['sequence'], custom_index=index)

        # save features
        features = self.features.keys() if features is None else features
        for feature_name, feature_data in self.features.items():
            if feature_name in features:
                feature_path = os.path.join(self.path, f'{feature_name}.h5')
                if not os.path.exists(feature_path) or overwrite:
                    check_path(feature_path)
                    with h5py.File(feature_path, 'w') as f:
                        f.create_dataset(feature_name, data=feature_data)
            else:
                pass  # do not save the feature

    def load_metadata(self, indices=None):
        if os.path.exists(self.csv):
            self.metadata = pd.read_csv(self.csv)
            if indices is not None:
                self.metadata = self.metadata.iloc[indices].reset_index(drop=True)
        else:
            raise FileNotFoundError(f'File {self.csv} not found')

        for feature_name in os.listdir(self.path):
            if feature_name.endswith('.h5'):
                self.features[feature_name[:-3]] = os.path.join(self.path, feature_name)

    def load_features(self, indices=None, features=None):
        features = self.features.keys() if features is None else features
        for feature_name, feature_path in self.features.items():
            if feature_name in features and isinstance(feature_path, str) and feature_path.endswith('.h5'):
                with h5py.File(feature_path, 'r') as f:
                    self.features[feature_name] = f[feature_name][()] if indices is None else f[feature_name][indices]
                self.feature_state[feature_name] = True

    def load(self, indices=None, features=None):
        # indices: the indices of the samples to be loaded, including metadata and features
        self.load_metadata(indices)
        self.load_features(indices, features)

    def split(self, test_size=0.2, valid_size=None, stratify=None, mode=None, q=None, seed=42, **kwargs):
        # print('test_size:', test_size)
        # print('valid_size:', valid_size)
        # print('stratify:', stratify)
        # print('mode:', mode)
        # print('q:', q)
        # print('kwargs:', kwargs)
        stratify_col = 'length' if stratify is None else stratify

        if mode is None:  # auto mode detection
            if pd.api.types.is_integer_dtype(self.metadata[stratify_col]):
                mode = 'discrete'
            elif pd.api.types.is_string_dtype(self.metadata[stratify_col]):
                mode = 'discrete'
            elif pd.api.types.is_numeric_dtype(self.metadata[stratify_col]):
                mode = 'continuous'
            else:
                raise ValueError(f'Unsupported data type {self.metadata[stratify].dtype}')

        if mode == 'continuous':
            stratify_col = stratify_col + '_bin'
            q = len(self.metadata) // 10 if q is None else q
            self.metadata[stratify_col] = pd.qcut(self.metadata[stratify], q=q, **kwargs)
        else:
            pass

        # 分层划分训练集和测试集
        if test_size > 0:
            no_nan_indices = self.metadata[~self.metadata[stratify_col].isnull()].index
            stratify_values = self.metadata.loc[no_nan_indices, stratify_col]

            train_indices, test_indices = train_test_split(
                no_nan_indices,
                test_size=test_size,
                random_state=seed,
                stratify=stratify_values
            )
            self.metadata.loc[train_indices, 'split'] = 'train'
            self.metadata.loc[test_indices, 'split'] = 'test'
        else:
            train_indices = self.metadata[~self.metadata[stratify_col].isnull()].index
            self.metadata['split'] = 'train'

        # 如果有验证集大小定义，则进一步划分训练集和验证集
        if valid_size is not None:
            train_indices, valid_indices = train_test_split(
                train_indices,
                test_size=valid_size,
                random_state=seed,
                stratify=self.metadata.loc[train_indices, stratify_col]
            )
            self.metadata.loc[valid_indices, 'split'] = 'valid'

    def construct_subset(self, indices, name=None):
        subset = ProteinDataset(name=name, path=self.path)
        subset.metadata = self.metadata.loc[indices].reset_index(drop=True)
        for feature_name, feature_data in self.features.items():
            if isinstance(feature_data, (np.ndarray, torch.Tensor)):
                feature_data = feature_data[indices]
            elif isinstance(feature_data, list):
                feature_data = [feature_data[i] for i in indices]
            elif isinstance(feature_data, str) and feature_data.endswith('.h5'):  # lazy loading
                subset.h5_index_map = {i: j for i, j in enumerate(indices)}  # map the index of the subset to the original dataset
            else:
                raise ValueError(f'Unsupported feature data type {type(feature_data)}')
            subset.features[feature_name] = feature_data
        return subset

    def get_indices(self, split):
        return self.metadata[self.metadata['split'] == split].index.tolist()

    def run_wrapper(self, name, wrapper_class, result_type='metadata', result_key=None, **kwargs):
        assert result_type in ['metadata', 'features'], 'Unsupported result type'
        wrapper = wrapper_class(output_dir=self.path, **kwargs)
        self.__setattr__(name, wrapper)
        assert callable(wrapper), 'Wrapper class should be callable'
        result = wrapper(self, **kwargs)
        result_key = name if result_key is None else result_key  # the key of the result
        if result_type == 'metadata':
            self.add_metadata(result, column_name=result_key, map_key='sequence', **kwargs)
        elif result_type == 'features':
            self.add_feature({result_key: result})
        else:
            raise ValueError(f'Unsupported result type {result_type}')

    def __len__(self):
        return len(self.metadata)

    def __str__(self):
        return f'ProteinDataset[ {self.name} ], size: {self.__len__()}, path: {self.path}'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index):
        item = {}
        for column_name in self.metadata.columns:
            item[column_name] = self.metadata[column_name][index]
        for feature_name, feature_data in self.features.items():
            if self.feature_state[feature_name]:  # eager loading
                item[feature_name] = feature_data[index]
            else:  # lazy loading
                with h5py.File(feature_data, 'r') as f:
                    item[feature_name] = f[feature_name][self.h5_index_map[index]]
        return item
