import esm
import numpy as np
import torch
from aaindex import aaindex1
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ProteinTokenizer:
    '''
    ESM-1b alphabet:
    ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E',
        'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W',
        'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
    '''

    def __init__(self, **kwargs):
        alphabet = kwargs.get('alphabet', 'ESM-1b')  # 默认使用ESM-1b的字母表
        if alphabet != 'ESM-1b':
            raise NotImplementedError

        truncation_seq_length = kwargs.get('truncation_seq_length', None)  # 超过该长度的序列会被截断
        no_gap = kwargs.get('no_gap', True)  # 是否去除gap字符

        self.alphabet, self.tokenizer = self.init_tokenizer(alphabet, truncation_seq_length)
        self.alphabet_size = len(self.alphabet)
        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v: k for k, v in self.tok_to_idx.items()}

        self.standard_cols = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                              14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.special_token_cols = [0, 1, 2, 3]
        self.special_residue_cols = [24, 25, 26, 27, 28]
        self.gap_col = [30]

        self.aaindex = AAIndex(no_gap=no_gap)

    def init_tokenizer(self, alphabet, truncation_seq_length):
        alphabet = esm.data.Alphabet.from_architecture(alphabet)
        batch_converter = alphabet.get_batch_converter(truncation_seq_length)
        return alphabet, batch_converter

    def tokenize(self, sequences):
        sequences = [(0, sequences)] if isinstance(sequences, str) else sequences  # str -> [(0, str)]
        sequences = [x if isinstance(x, tuple) else (i, x) for i, x in enumerate(sequences)]  # [(index, sequence)]
        batch_ids, batch_seqs, batch_tokens = self.tokenizer(sequences)  # (N, ), (N, L), (N, L), L = max_seq_len+2
        return batch_ids, batch_seqs, batch_tokens

    def ids2sequences(self, batch_indices):
        batch_indices = batch_indices.unsqueeze(0) if len(batch_indices.shape) == 1 else batch_indices  # (seq_len, ) -> (1, seq_len)

        mask = torch.ones_like(batch_indices, dtype=torch.bool)
        for element in self.special_token_cols:
            mask &= batch_indices != element

        flatten_indices = batch_indices[mask]
        flatten_residues = [self.idx_to_tok[idx.item()] for idx in flatten_indices]
        sequences = []
        for seq_len in mask.sum(dim=1):
            sequence = ''.join(flatten_residues[:seq_len])
            flatten_residues = flatten_residues[seq_len:]
            sequences.append(sequence)
        return sequences

    def encode_one_hot(self, sequences, padding=True, special_token=False, special_residue=False, include_gap=False):
        sequences = [sequences] if isinstance(sequences, str) else sequences  # str -> [str]
        _, _, batch_tokens = self.tokenize(sequences)
        one_hot_encodings = torch.nn.functional.one_hot(batch_tokens[:, 1:-1], num_classes=self.alphabet_size).long()  # (N, max_seq_len, 33)

        # 如果只选4-23列会使得只有20个标准氨基酸的onehot编码是正常的，而其他的特殊符号对应的onehot编码为全0
        select_cols = self.standard_cols.copy()
        if special_token:
            select_cols += self.special_token_cols
        if special_residue:
            select_cols += self.special_residue_cols
        if include_gap:
            select_cols += self.gap_col
        select_cols = torch.tensor(select_cols, dtype=torch.long, device=one_hot_encodings.device)
        one_hot_encodings = one_hot_encodings[:, :, select_cols]  # (N, max_seq_len, x>=20)

        if not padding:  # 如果不需要padding，则将onehot编码的长度截断到与序列长度一致
            truncated_encodings = []  # 变长list，每个元素是一个(N, seq_len, 20)的onehot编码
            for sequence, one_hot_encoding in zip(sequences, one_hot_encodings):
                one_hot_encoding = one_hot_encoding[:len(sequence)]
                truncated_encodings.append(one_hot_encoding)
            one_hot_encodings = truncated_encodings
        return one_hot_encodings

    def encode_aaindex(self, sequences, flatten=True, normalize=None):
        return self.aaindex.batch_encode(sequences, flatten, normalize)


class AAIndex:
    def __init__(self, no_gap=True):
        super(AAIndex, self).__init__()
        amino_acids = aaindex1.amino_acids()
        if no_gap:
            amino_acids.remove('-')
        self.one_hot_array = np.eye(len(amino_acids))
        self.aa_row_index, self.aaindex_array = self.get_aaindex_table(amino_acids)

    def get_aaindex_table(self, amino_acids):
        aa_row_index = {a: i for i, a in enumerate(amino_acids)}
        aaindex_array = np.zeros([len(aa_row_index), aaindex1.num_records()])
        col_index = 0
        for i, x in aaindex1.parse_aaindex().items():
            # print(f'{i}: {x}')
            for a, v in x['values'].items():
                if a in aa_row_index:
                    aaindex_array[aa_row_index[a], col_index] = v
                else:
                    if a != '-':
                        raise Warning()
            col_index += 1
        return aa_row_index, aaindex_array

    def batch_encode(self, sequences, flatten=True, normalize=None):
        sequences = [sequences] if isinstance(sequences, str) else sequences  # str -> [str]
        id_list = [[self.aa_row_index[aa] for aa in seq] for seq in sequences]  # (N, seq_len)
        aaindex_encoding = [self.aaindex_array[ids] for ids in id_list]  # (N, seq_len, 566)
        # pad aaindex_encoding to the same length with 0
        max_len = max([len(ids) for ids in id_list])
        pad_aaindex_encoding = []
        for encoding in aaindex_encoding:
            pad_aaindex_encoding.append(np.pad(encoding, ((0, max_len - len(encoding)), (0, 0)), 'constant', constant_values=0))
        aaindex_encoding = np.array(pad_aaindex_encoding)  # (N, max_seq_len, 566)

        if flatten:
            aaindex_encoding = aaindex_encoding.reshape(aaindex_encoding.shape[0], -1)
            if normalize:
                if normalize == 'standard':
                    scalar = StandardScaler()
                elif normalize == 'standard':
                    scalar = MinMaxScaler()
                else:
                    raise NotImplementedError
                aaindex_encoding = scalar.fit_transform(aaindex_encoding)
        return aaindex_encoding
