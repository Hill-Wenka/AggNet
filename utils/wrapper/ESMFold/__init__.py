import os

import esm
import pandas as pd
import torch
# from tqdm.notebook import tqdm
from tqdm import tqdm

from ... import conda_path, root_path
from ...bio import get_atom_array
from ...datastructure import merge_dicts
from ...file import check_path, is_path_exist, list_file, read_fasta, write_fasta
from ...parallel import concurrent_submit

shell_script = os.path.join(root_path, 'utils', 'wrapper', 'ESMFold', 'run.sh')
python_script = os.path.join(root_path, 'utils', 'wrapper', 'ESMFold', 'esmfold_inference.py')
esmfold_env = 'esmfold'


def get_command(input, output, max_tokens_per_batch, num_recycles, cpu_only, cpu_offload):
    return f'sh {shell_script} {python_script} {input} {output} {max_tokens_per_batch} {num_recycles} {cpu_only} {cpu_offload} {conda_path} {esmfold_env}'


def run_script(fasta, output_dir, max_tokens_per_batch, num_recycles=5, cpu_only=False, cpu_offload=False):
    command = get_command(fasta, output_dir, max_tokens_per_batch, num_recycles, cpu_only, cpu_offload)
    return os.system(command)


def parallel_load_atom_array(index, pdb_file):
    try:  # must use try except, otherwise the program will be terminated if one of the structure is invalid
        structure = get_atom_array(pdb_file)
    except:
        print(f'[ESMFold] Error: {index} failed to load')
        structure = None
    return {index: structure}


class ESMFoldWrapper:
    def __init__(
            self,
            output_dir=None,
            num_recycles=5,
            cpu_only=False,
            cpu_offload=False,
            max_tokens_per_batch=1000,
            chunk_size=None,
            device=None,
            **kwargs
    ):
        if output_dir is None:
            output_dir = './esmfold/'
            # print(f'Output directory is not provided, set to default: {output_dir}')
        else:
            output_dir = os.path.join(output_dir, 'esmfold/')

        self.output_dir = os.path.abspath(output_dir)  # relative path will not work
        self.num_recycles = num_recycles
        self.cpu_only = cpu_only
        self.cpu_offload = cpu_offload
        self.max_tokens_per_batch = max_tokens_per_batch
        self.chunk_size = chunk_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu_only else 'cpu') if device is None else torch.device(device)

        self.temp_fasta = os.path.join(self.output_dir, 'esmfold_temp.fasta')
        self.temp_df = os.path.join(self.output_dir, 'temp_df.csv')
        self.result_file = os.path.join(self.output_dir, 'esmfold_result.csv')
        self.pdb_dir = os.path.join(self.output_dir, 'pdf_files/')
        self.result = None

        self.model = None

    def __init_submodule__(self):
        print('[ESM] ESM model initializing...')
        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        if self.chunk_size is not None:
            self.model.set_chunk_size(self.chunk_size)

    def compute(self, sequences, overwrite=False):
        check_path(self.output_dir)
        check_path(self.pdb_dir)

        if self.cpu_only:  # only run on cpu
            num_finished, num_total, temp_seqs, state = self.prepare(sequences, overwrite)
            max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
            print(f'[ESMFold] Number of finished before cpu running: {num_finished}/{num_total}, max_len: {max_len}')
            result = 0 if state else run_script(self.temp_fasta,
                                                self.pdb_dir,
                                                self.max_tokens_per_batch,
                                                self.num_recycles,
                                                True,
                                                self.cpu_offload)
        else:
            # try gpu first, then run on cpu
            num_finished, num_total, temp_seqs, state = self.prepare(sequences, overwrite)
            max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
            print(f'[ESMFold] Number of finished before gpu running: {num_finished}/{num_total}, max_len: {max_len}')
            result = 0 if state else run_script(self.temp_fasta,
                                                self.pdb_dir,
                                                self.max_tokens_per_batch,
                                                self.num_recycles,
                                                False,
                                                self.cpu_offload)

            if result == 0:
                seqs, heads = read_fasta(self.temp_fasta)
                pdb_files = [os.path.join(self.pdb_dir, head + '.pdb') for head in heads]
                df = pd.DataFrame({'sequence': seqs, 'pdb_file': pdb_files})
                self.result = pd.concat([self.result, df], ignore_index=True)
                self.result.drop_duplicates(subset=['sequence'], keep='last', inplace=True)
                self.result.to_csv(self.result_file, index=False)
            else:
                raise RuntimeError('[ESMFold] Prediction failed')

            # some sequences failed to run on gpu (e.g. out of memory), need to run on cpu
            num_finished, num_total, temp_seqs, state = self.prepare(sequences, overwrite)
            max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
            print(f'[ESMFold] Number of finished before cpu running: {num_finished}/{num_total}, max_len: {max_len}')
            result = 0 if state else run_script(self.temp_fasta,
                                                self.pdb_dir,
                                                self.max_tokens_per_batch,
                                                self.num_recycles,
                                                True,
                                                self.cpu_offload)

        if result == 0:
            seqs, heads = read_fasta(self.temp_fasta)
            pdb_files = [os.path.join(self.pdb_dir, head + '.pdb') for head in heads]
            df = pd.DataFrame({'sequence': seqs, 'pdb_file': pdb_files})
            self.result = pd.concat([self.result, df], ignore_index=True)
            self.result.drop_duplicates(subset=['sequence'], keep='last', inplace=True)
            self.result.to_csv(self.result_file, index=False)
        else:
            raise RuntimeError('[ESMFold] Prediction failed')
        return result

    def prepare(self, sequences, overwrite=False):
        self.check_data()  # update the result file with the finished data
        self.result = pd.read_csv(self.result_file) if is_path_exist(self.result_file) else pd.DataFrame(columns=['sequence', 'pdb_file'])
        self.result['sequence'] = self.result['sequence'].astype(str)
        temp_seqs = list(sequences) if overwrite else list(set(sequences) - set(self.result['sequence'].values))
        temp_indices = [f'seq_{i}' for i in range(len(self.result), len(self.result) + len(temp_seqs))]
        check_path(self.temp_fasta)
        write_fasta(self.temp_fasta, temp_seqs, temp_indices)
        state = True if len(temp_seqs) == 0 else False

        # save checkpoint records
        temp_pdb_files = [os.path.join(self.pdb_dir, index + '.pdb') for index in temp_indices]
        temp_df = pd.DataFrame({'sequence': temp_seqs, 'pdb_file': temp_pdb_files})
        temp_df.to_csv(self.temp_df, index=False)
        return len(sequences) - len(temp_seqs), len(sequences), temp_seqs, state

    def load_data(self, sequences=None, parallel=False, atom_array=True, **kwargs):
        result_df = pd.read_csv(self.result_file)
        load_seqs = result_df['sequence'].values if sequences is None else sequences
        pdb_files = result_df.set_index('sequence').loc[load_seqs]['pdb_file']  # keep the order of load_seqs unchanged

        if atom_array:
            if parallel:  # 不一定比串行快
                params = [(i, pdb_file) for i, pdb_file in enumerate(pdb_files)]
                results = concurrent_submit(parallel_load_atom_array, params, desc='Loading ESMFold data')
                results = merge_dicts(results)
                results = [results[i] for i in range(len(results))]
            else:
                results = [get_atom_array(pdb_file) for pdb_file in tqdm(pdb_files, desc='Loading ESMFold data')]
        else:
            results = result_df
        return results

    def __call__(self, dataset, **kwargs):
        # interface for dataset to call, dataset should have 'sequence' column
        unique_sequences = dataset.df['sequence'].unique()
        print('[ESMFold] Computing ESMFold prediction, unique sequences:', len(unique_sequences))
        result = self.compute(unique_sequences)
        assert result == 0, '[ESMFold] Prediction failed'
        return self.load_data(sequences=dataset.sequences, **kwargs)  # ensure the order of the sequences consistent with the dataset

    # need to install and activate the esmfold environment
    def forward(self, sequences):
        if self.model is None:
            self.__init_submodule__()
        check_path(self.output_dir)
        data = [(f'seq_{i}', seq) for i, seq in enumerate(sequences)]

        with torch.no_grad():
            for index, sequence in tqdm(data, desc='Predicting ESMFold structure'):
                output = self.model.infer_pdb(sequence)
                with open(os.path.join(self.pdb_dir, index + '.pdb'), "w") as f:
                    f.write(output)

        result = {}
        indices, pdb_files, atom_arrays = [], [], []
        for index, sequence in data:
            pdb_file = os.path.join(self.pdb_dir, index + '.pdb')
            structure = get_atom_array(pdb_file)
            indices.append(index)
            pdb_files.append(pdb_file)
            atom_arrays.append(structure)

        result['dataframe'] = pd.DataFrame({'index': indices, 'sequence': sequences, 'pdb_file': pdb_files})
        result['atom_array'] = atom_arrays
        return result

    def __repr__(self):
        return f'ESMFoldWrapper(path={self.output_dir})'

    def check_data(self):
        self.result = pd.read_csv(self.result_file) if is_path_exist(self.result_file) else pd.DataFrame(columns=['sequence', 'pdb_file'])
        temp_df = pd.read_csv(self.temp_df) if is_path_exist(self.temp_df) else pd.DataFrame(columns=['sequence', 'pdb_file'])

        finished_pdbs = list_file(self.pdb_dir, absolute=True)  # all finished pdb files in the pdb_dir
        finished_pdbs = [f for f in finished_pdbs if f.endswith('.pdb')]  # only keep pdb files
        temp_df.set_index('pdb_file', inplace=True)
        finished_data = [(temp_df.loc[pdb, 'sequence'], pdb) for pdb in finished_pdbs if pdb in temp_df.index]  # all finished data
        if len(finished_data) > 0:
            finished_seqs, finished_pdbs = zip(*finished_data)
            df = pd.DataFrame({'sequence': finished_seqs, 'pdb_file': finished_pdbs})
        else:
            df = pd.DataFrame(columns=['sequence', 'pdb_file'])

        # update the result file with the finished data
        self.result = pd.concat([self.result, df], ignore_index=True)
        self.result.drop_duplicates(subset=['sequence'], keep='last', inplace=True)
        self.result = self.result.sort_values(by='pdb_file').reset_index(drop=True)
        self.result.to_csv(self.result_file, index=False)
