import os

import esm
import pandas as pd
import torch
from tqdm.notebook import tqdm

from ... import conda_env, conda_path, root_path
from ...datastructure import merge_dicts
from ...file import check_path, is_path_exist, read_fasta, write_fasta
from ...parallel import concurrent_submit

esm_model_names = ['esm1_t34_670M_UR50S',
                   'esm1_t34_670M_UR50D',
                   'esm1_t34_670M_UR100',
                   'esm1_t12_85M_UR50S',
                   'esm1_t6_43M_UR50S',
                   'esm1b_t33_650M_UR50S',
                   'esm_msa1_t12_100M_UR50S',
                   'esm_msa1b_t12_100M_UR50S',
                   'esm1v_t33_650M_UR90S_1',
                   'esm1v_t33_650M_UR90S_2',
                   'esm1v_t33_650M_UR90S_3',
                   'esm1v_t33_650M_UR90S_4',
                   'esm1v_t33_650M_UR90S_5',
                   'esm_if1_gvp4_t16_142M_UR50',
                   'esm2_t6_8M_UR50D',
                   'esm2_t12_35M_UR50D',
                   'esm2_t30_150M_UR50D',
                   'esm2_t33_650M_UR50D',
                   'esm2_t36_3B_UR50D',
                   'esm2_t48_15B_UR50D']

shell_script = os.path.join(root_path, 'utils', 'wrapper', 'ESM', 'run.sh')
python_script = os.path.join(root_path, 'utils', 'wrapper', 'ESM', 'extract.py')


def get_command(input, output, model_name, repr_layers, include, nogpu):
    return f'sh {shell_script} {python_script} {model_name} {input} {output} {repr_layers} {include} {nogpu} {conda_path} {conda_env}'


def run_script(fasta, output_dir, model_name, repr_layers, include, nogpu):
    return os.system(get_command(fasta, output_dir, model_name, repr_layers, include, nogpu))


def parallel_load_tensor(index, pt_file):
    try:
        data = torch.load(pt_file, weights_only=False)
    except:
        print(f'[ESM] Error: {index} failed to load')
        data = None
    return {index: data}


class ESMWrapper:
    def __init__(
            self,
            output_dir=None,
            model_name='esm2_t33_650M_UR50D',
            repr_layers='33',
            include='per_tok',
            nogpu=False,
            device=None,
            **kwargs
    ):
        assert model_name in esm_model_names

        if output_dir is None:
            output_dir = f'./{model_name}/'
            # print(f'Output directory is not provided, set to default: {output_dir}')
        else:
            output_dir = os.path.join(output_dir, f'{model_name}/')

        self.output_dir = os.path.abspath(output_dir)  # relative path will not work
        self.model_name = model_name
        self.repr_layers = repr_layers
        self.include = include  # logits, mean, per_tok, contacts
        self.nogpu = nogpu
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu') if device is None else torch.device(device)

        self.layer_list = [int(self.repr_layers)] if isinstance(self.repr_layers, (int, str)) else [int(x) for x in
                                                                                                    self.repr_layers]
        self.return_contacts = 'contacts' in self.include

        self.temp_fasta = os.path.join(self.output_dir, 'esm_temp.fasta')
        self.result_file = os.path.join(self.output_dir, 'esm_result.csv')
        self.tensor_dir = os.path.join(self.output_dir, 'tensor_files')
        self.result = None

        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def __init_submodule__(self):
        print('[ESM] ESM model initializing...')
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model = self.model.to(self.device)
        self.standard_aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                             'Y', 'V']
        self.standard_aa_indices = [self.alphabet.tok_to_idx[aa] for aa in self.standard_aas]

    def compute(self, sequences, overwrite=False):
        check_path(self.output_dir)
        num_finished, num_total, temp_seqs, state = self.prepare(sequences, overwrite)
        max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
        print(f'[ESM] Number of finished before running: {num_finished}/{num_total}, max_len: {max_len}')
        result = 0 if state else run_script(self.temp_fasta, self.tensor_dir, self.model_name, self.repr_layers,
                                            self.include, self.nogpu)

        if result == 0:
            seqs, heads = read_fasta(self.temp_fasta)
            pt_files = [os.path.join(self.tensor_dir, head + '.pt') for head in heads]
            df = pd.DataFrame({'sequence': seqs, 'pt_file': pt_files})
            self.result = pd.concat([self.result, df], ignore_index=True)
            self.result.drop_duplicates(subset=['sequence'], keep='last', inplace=True)
            self.result.to_csv(self.result_file, index=False)
        else:
            raise RuntimeError('[ESM] Prediction failed')
        return result

    def prepare(self, sequences, overwrite=False):
        self.result = pd.read_csv(self.result_file) if is_path_exist(self.result_file) else pd.DataFrame(
            columns=['sequence', 'pt_file'])
        self.result['sequence'] = self.result['sequence'].astype(str)
        temp_seqs = list(sequences) if overwrite else list(set(sequences) - set(self.result['sequence'].values))
        temp_indices = [f'seq_{i}' for i in range(len(self.result), len(self.result) + len(temp_seqs))]
        check_path(self.temp_fasta)
        write_fasta(self.temp_fasta, temp_seqs, temp_indices)
        state = True if len(temp_seqs) == 0 else False
        return len(sequences) - len(temp_seqs), len(sequences), temp_seqs, state

    def load_data(self, sequences=None, key='representations', layer=33, mean=False, parallel=False, **kwargs):
        result_df = pd.read_csv(self.result_file)
        load_seqs = result_df['sequence'].values if sequences is None else sequences
        pt_files = result_df.set_index('sequence').loc[load_seqs]['pt_file']  # keep the order of load_seqs unchanged

        if parallel:  # 不一定比串行快
            params = [(i, pt_file) for i, pt_file in enumerate(pt_files)]
            results = concurrent_submit(parallel_load_tensor, params, desc='Loading ESM data')
            results = merge_dicts(results)
            results = [results[i] for i in range(len(results))]
        else:
            results = [torch.load(pt_file, weights_only=False) for pt_file in tqdm(pt_files, desc='Loading ESM data')]

        features = []
        for seq, data in zip(load_seqs, results):
            if key == 'representations':
                feature = data[key][layer]  # 已经是cpu tensor并且已经去掉了cls, eos, padding
            elif key == 'mean_representations':
                feature = data[key][layer]
            elif key == 'contacts':
                feature = data[key]
            elif key == 'logits':
                feature = data[key]
            else:
                raise ValueError(f'Unsupported key: {key}')

            assert feature.shape[0] == len(seq), f'length of the sequence does not match the feature: {seq}'

            if key == 'representations' and mean:
                feature = feature.mean(0)
            features.append(feature)
        return torch.stack(features)

    def __call__(self, dataset, **kwargs):
        # interface for dataset to call, dataset should have a 'sequence' column
        unique_sequences = dataset.df['sequence'].unique()
        print('[ESM] Computing ESM prediction, unique sequences:', len(unique_sequences))
        result = self.compute(unique_sequences)
        assert result == 0, '[ESM] Prediction failed'
        return self.load_data(sequences=dataset.sequences,
                              **kwargs)  # ensure the order of the sequences consistent with the dataset

    def forward(self, sequences):
        if self.model is None:
            self.__init_submodule__()
        data = [(f'seq_{i}', seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device), repr_layers=self.layer_list,
                                 return_contacts=self.return_contacts)

        mean_representations_list = []
        logits_list = []
        contacts_list = []
        for i, (index, seq) in enumerate(data):
            if 'logits' in self.include:
                logits = results['logits'][i, 1: len(seq) + 1].cpu()  # (N, L, 33)
                logits_list.append(logits)
            if 'mean' in self.include:
                mean_representations = {}
                for key, value in results['representations'].items():
                    mean_representations[key] = value[i, 1: len(seq) + 1].mean(0).cpu()
                mean_representations = mean_representations[self.layer_list[-1]] if len(
                    self.layer_list) == 1 else mean_representations
                mean_representations_list.append(mean_representations)
            if 'contacts' in self.include:
                contacts = results['contacts'][i].cpu()
                contacts_list.append(contacts)

        features = {}
        features['representations'] = results['representations'][self.layer_list[-1]] if len(self.layer_list) == 1 else \
        results['representations']
        if len(mean_representations_list) > 0:
            features['mean_representations'] = torch.stack(mean_representations_list)
        if len(logits_list) > 0:
            features['logits'] = torch.stack(logits_list)
        if len(contacts_list) > 0:
            features['contacts'] = torch.stack(contacts_list)
        return features

    def scan_sequence(self, sequence, batch_size=64):
        # replace each amino acid in the sequence with <MASK> and predict the probability of each amino acid
        if self.model is None:
            self.__init_submodule__()

        mutated_sequences = [sequence[:i] + '<mask>' + sequence[i + 1:] for i in range(len(sequence))]
        probabilities = torch.empty((len(sequence), 20))
        for i in tqdm(range(0, len(mutated_sequences), batch_size), desc='Scanning sequence'):
            data = [(f'mask_{i + j + 1}', seq) for j, seq in enumerate(mutated_sequences[i: i + batch_size])]
            _, _, batch_tokens = self.batch_converter(data)
            with torch.no_grad():
                results = self.model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=False)
            logits = results['logits'][:, 1:len(sequence) + 1].cpu()
            for j in range(len(data)):
                probabilities[i + j] = logits[j, i + j, self.standard_aa_indices].softmax(dim=-1)
                assert batch_tokens[:, 1:-1][j, i + j] == self.alphabet.tok_to_idx[
                    '<mask>'], f'Error: {batch_tokens[:, 1:-1][j, i + j]}'

        scan_probs_df = pd.DataFrame(probabilities, columns=self.standard_aas)
        return scan_probs_df

    def __repr__(self):
        return f'ESMWrapper(path={self.output_dir})'
