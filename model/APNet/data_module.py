import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.dataset import ProteinDataset
from utils.dataset.Tokenizer import ProteinTokenizer
from utils.wrapper import ESM


def constrcut_dataset(peptides):
    pred_dataset = ProteinDataset('./pred_dataset', sequence=peptides)
    df = pred_dataset.metadata
    pred_dataset.run_wrapper('esm',
                             ESM.ESMWrapper,
                             include='per_tok',
                             repr_layers='33',
                             result_key='esm_embedding',
                             result_type='features')
    tokenizer = ProteinTokenizer()
    onehots = tokenizer.encode_one_hot(df.sequence)
    aaindex = tokenizer.aaindex.batch_encode(df.sequence)
    pred_dataset.add_feature({'onehot': onehots, 'aaindex': aaindex})
    return pred_dataset


class DataModule(L.LightningDataModule):
    def __init__(self, config, log=True):
        super().__init__()

        self.log = log
        self.config = config
        self.dataset_args = self.config.dataset
        self.tokenization_args = self.config.tokenization
        self.train_dataloader_args = self.config.train_dataloader
        self.valid_dataloader_args = self.config.valid_dataloader
        self.test_dataloader_args = self.config.test_dataloader
        self.predict_dataloader_args = self.config.predict_dataloader

        dataset_args = self.dataset_args.dataset
        self.dataset = ProteinDataset(dataset_args) if isinstance(dataset_args, str) else dataset_args
        self.tokenizer = ProteinTokenizer(**self.tokenization_args)
        self.select_indices = None
        self.dataframe = None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.max_len = self.dataset_args.max_len if self.dataset_args.max_len is not None else self.dataset.metadata.length.max()
        self.state = False  # whether the data is prepared

    def prepare_data(self):
        if not self.state:  # run only once to avoid repeated preparation and split changes
            # each run gives different select_indices. Thus, ensure only one run to avoid different results
            # select data with sequence length <= max_len
            select_indices = self.dataset.metadata[self.dataset.metadata['length'] <= self.max_len].index

            # select a subset of the dataset for debug
            subset_ratio = self.dataset_args.mini_set_ratio if self.dataset_args.mini_set_ratio is not None else 1
            self.select_indices = pd.DataFrame(select_indices).sample(frac=subset_ratio, random_state=self.config.seed)[
                0].values
            self.dataset.metadata = self.dataset.metadata.loc[self.select_indices]

            # split the dataset into train, valid, test
            if self.dataset_args.split is not None:
                self.dataset.split(**self.dataset_args.split)
                if self.log:
                    print('split the dataset into train, valid, test')
            else:
                if self.log:
                    print('use the original split of the dataset')

            self.dataframe = self.dataset.metadata
            self.train_index = self.dataframe[self.dataframe['split'] == 'train'].index.tolist()
            self.valid_index = self.dataframe[self.dataframe['split'] == 'valid'].index.tolist()
            self.test_index = self.dataframe[self.dataframe['split'] == 'test'].index.tolist()
            self.test_index = self.test_index if len(self.test_index) > 0 else self.train_index
            self.valid_index = self.valid_index if len(self.valid_index) > 0 else self.test_index
            self.predict_index = None
            self.state = True

            if self.log:
                print(
                    f'[prepare_data] max_len: {self.max_len}, subset_ratio: {subset_ratio}, number: {len(self.dataframe)}')

            self.load_cache()

    def setup(self, stage=None):
        if self.log:
            print('=' * 30, f'Setup [{stage}] Start', '=' * 30)
        if stage == 'fit':
            self.train_dataset = self.dataset.construct_subset(self.train_index, 'train_dataset')
            self.valid_dataset = self.dataset.construct_subset(self.valid_index, 'valid_dataset')
            if self.log:
                print('[self.train_dataset]', len(self.train_dataset))
                print('[self.val_dataset]', len(self.valid_dataset))
        elif stage == 'validate':
            self.valid_dataset = self.dataset.construct_subset(self.valid_index, 'valid_dataset')
            if self.log:
                print('[self.val_dataset]', len(self.valid_dataset))
        elif stage == 'test':
            self.test_dataset = self.dataset.construct_subset(self.test_index, 'test_dataset')
            if self.log:
                print('[self.test_dataset]', len(self.test_dataset))
        elif stage == 'predict':
            if self.predict_dataset is None:
                self.predict_index = self.test_index if self.predict_index is None else self.predict_index
                self.predict_dataset = self.dataset.construct_subset(self.predict_index, 'predict_dataset')
            else:
                pass  # dataset has been provided when calling self.prepare_predict_data()
            if self.log:
                print('[self.predict_dataset]', len(self.predict_dataset))
        else:
            print('stage', stage)
            raise RuntimeError(f'Parameter {stage} is None or illegal, please set it properly')
        if self.log:
            print('=' * 30, f'Setup [{stage}] End', '=' * 30)

    def load_cache(self):
        if self.log:
            print('=' * 30, f'Loading Cache', '=' * 30)

        def load_and_convert_tensors(self, attribute_name, file_path):
            if file_path is not None:
                data = torch.load(file_path, weights_only=False)
                for key in data:
                    if isinstance(data[key], np.ndarray):
                        data[key] = torch.from_numpy(data[key])
                    elif not isinstance(data[key], torch.Tensor):
                        raise ValueError('Unsupported type for key {}: {}'.format(key, type(data[key])))
                setattr(self, attribute_name, data)
            else:
                setattr(self, attribute_name, None)

        load_and_convert_tensors(self, 'seq2esm', self.config.cache.seq2esm)
        load_and_convert_tensors(self, 'seq2token', self.config.cache.seq2token)
        load_and_convert_tensors(self, 'seq2aaindex', self.config.cache.seq2aaindex)
        self.selected_aaindex = torch.load(self.config.cache.selected_aaindex, weights_only=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.collate_fn, **self.train_dataloader_args)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, collate_fn=self.collate_fn, **self.valid_dataloader_args)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.collate_fn, **self.test_dataloader_args)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, collate_fn=self.collate_fn, **self.predict_dataloader_args)

    def collate_fn(self, batch):
        batch_labels = torch.tensor([item['label'] for item in batch], dtype=torch.int64)
        batch_seqs = [item['sequence'] for item in batch]
        batch_ids, batch_seqs, batch_tokens = self.tokenizer.tokenize(batch_seqs)
        batch_embeddings = None
        # batch_embeddings = torch.stack([self.seq2esm[seq] for seq in batch_seqs]).float() if self.seq2esm is not None else None
        # batch_aaindex = torch.stack(
        #     [self.seq2aaindex[seq] for seq in batch_seqs]).float() if self.seq2aaindex is not None else None
        batch_aaindex = self.tokenizer.aaindex.batch_encode(batch_seqs)[:, self.selected_aaindex]
        batch_aaindex = torch.from_numpy(batch_aaindex).float()

        batch = {
            'batch_labels': batch_labels,  # (B, T, E)
            'batch_seqs': batch_seqs,  # (B)
            'batch_tokens': batch_tokens,  # (B, T+2)
            'batch_embeddings': batch_embeddings,  # (B, T, 1280)
            'batch_aaindex': batch_aaindex  # (B, T, 300)
        }
        return batch

    def prepare_predict_data(self, sequence=None, dataset=None, subset=None, **kwargs):
        if sum([sequence is not None, dataset is not None, subset is not None]) != 1:
            raise ValueError('Please set one of the parameters: sequences, dataset, subset')

        if subset is not None:
            if not self.state:
                self.prepare_data()
            if subset == 'train':
                self.predict_index = self.train_index
            elif subset == 'valid':
                self.predict_index = self.valid_index
            elif subset == 'test':
                self.predict_index = self.test_index
            elif subset == 'train_valid':
                self.predict_index = self.train_index + self.valid_index
            elif subset == 'train_valid_test':
                self.predict_index = self.train_index + self.valid_index + self.test_index
            else:
                raise ValueError(f'subset: {subset} is not valid')
            self.predict_dataset = self.dataset.construct_subset(self.predict_index, 'predict_dataset')

        if sequence is not None:
            self.predict_dataset = constrcut_dataset(sequence)
            self.predict_dataset.metadata['label'] = 0

        if dataset is not None:
            self.predict_dataset = dataset
            seq2esm = kwargs.get('seq2esm', None)
            seq2token = kwargs.get('seq2token', None)
            seq2aaindex = kwargs.get('seq2aaindex', None)
            if seq2esm is not None:
                self.seq2esm = seq2esm
            if seq2token is not None:
                self.seq2token = seq2token
            if seq2aaindex is not None:
                self.seq2aaindex = seq2aaindex

        if self.log:
            print('[prepare custom predict dataset]', len(self.predict_dataset))

        self.selected_aaindex =  torch.load('./checkpoint/selected_aaindex.pt', weights_only=False)

