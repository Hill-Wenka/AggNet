import os

import pandas as pd

from ... import conda_path
from ...datastructure import merge_dicts
from ...file import check_path, get_basename, is_path_exist
from ...parallel import concurrent_submit

'''
install aggrescan3d following the instruction in https://bitbucket.org/lcbio/aggrescan3d/src/master/
'''
a3d_env = 'aggrescan3d'


def get_command(input, output):
    return f'source {conda_path}/bin/activate {a3d_env}\naggrescan -i {input} -w {output} -v 0'


def run_script(input_pdb, output_dir):
    result_dir = os.path.join(output_dir, get_basename(input_pdb), '')
    script_file = os.path.join(result_dir, 'run.sh')
    assert is_path_exist(input_pdb), f'pdb file {input_pdb} does not exist'

    check_path(result_dir, log=False)
    with open(script_file, 'w') as f:
        f.write(get_command(input_pdb, result_dir))

    result = os.system(f'sh {script_file}')
    return result


def parallel_load_A3D_result(input_pdb, output_dir):
    index = get_basename(input_pdb)
    result_dir = os.path.join(output_dir, index, '')
    result_file = os.path.join(result_dir, 'A3D.csv')
    A3D_df = pd.read_csv(result_file) if is_path_exist(result_file) else None
    A3D_scores = A3D_df['score'].values if A3D_df is not None else None
    A3D_avg = A3D_scores.mean() if A3D_scores is not None else None
    sequence = ''.join(A3D_df['residue_name'].values) if A3D_df is not None else None
    return {index: (sequence, str(A3D_scores.tolist()), A3D_avg)}


class Aggrescan3DWrapper:
    def __init__(
            self,
            output_dir=None,
            **kwargs
    ):
        if output_dir is None:
            output_dir = './aggrescan3d/'
            # print(f'Output directory is not provided, set to default: {output_dir}')
        else:
            output_dir = os.path.join(output_dir, 'aggrescan3d/')

        self.output_dir = os.path.abspath(output_dir)  # relative path will not work
        self.result_dir = os.path.join(self.output_dir, 'result/')
        self.result_file = os.path.join(self.output_dir, 'a3d_result.csv')

    def compute(self, pdb_files, overwrite=False):
        check_path(self.output_dir)
        num_finished, num_total, temp_pdbs, state = self.prepare(pdb_files, overwrite)
        print(f'[Aggrescan3D] Number of finished before running: {num_finished}/{num_total}')
        params = [(pdb, self.result_dir) for pdb in temp_pdbs]
        if state:
            result = 0
        else:
            result = concurrent_submit(run_script, params, desc='Predicting A3D data')
            result = 0 if set(result) == {0} else 1

        if result == 0:
            params = [(pdb, self.result_dir) for pdb in pdb_files]
            A3D_scores = concurrent_submit(parallel_load_A3D_result, params, desc='Loading A3D data')
            A3D_scores = merge_dicts(A3D_scores)
            indices, sequences, A3D_scores_list, A3D_avg_scores = [], [], [], []
            for pdb in temp_pdbs:
                index = get_basename(pdb)
                sequence, scores, avg_scores = A3D_scores[index]
                indices.append(index)
                sequences.append(sequence)
                A3D_scores_list.append(scores)
                A3D_avg_scores.append(avg_scores)
            df = pd.DataFrame({'index': indices, 'pdb_sequence': sequences, 'pdb_file': temp_pdbs, 'a3d_avg_score': A3D_avg_scores,
                               'a3d_score': A3D_scores_list})
            self.result = pd.concat([self.result, df], ignore_index=True)
            self.result.drop_duplicates(subset=['pdb_file'], keep='last', inplace=True)
            self.result.to_csv(self.result_file, index=False)
        else:
            raise RuntimeError('[Aggrescan3D] prediction failed')
        return result

    def prepare(self, files, overwrite=False):
        self.result = pd.read_csv(self.result_file) if is_path_exist(self.result_file) else pd.DataFrame(
            columns=['index', 'pdb_sequence', 'pdb_file', 'a3d_avg_score', 'a3d_score'])
        self.result['pdb_file'] = self.result['pdb_file'].astype(str)
        temp_pdbs = list(files) if overwrite else list(set(files) - set(self.result['pdb_file'].values))
        state = True if len(temp_pdbs) == 0 else False
        return len(files) - len(temp_pdbs), len(files), temp_pdbs, state

    def load_data(self, **kwargs):
        df = pd.read_csv(self.result_file)
        df.drop(columns=['index'], inplace=True)
        return df

    def __call__(self, dataset, **kwargs):
        # interface for dataset to call, dataset should have a 'structure' column
        assert 'structure' in dataset.df.columns, 'structure column not found in dataset'
        unique_pdbs = dataset.df['structure'].unique()
        print('[Aggrescan3D] Computing Aggrescan3D prediction, unique structure:', len(unique_pdbs))
        result = self.compute(unique_pdbs)
        assert result == 0, '[Aggrescan3D] Prediction failed'
        return self.load_data(**kwargs)

    def __repr__(self):
        return f'Aggrescan3DWrapper(path={self.output_dir})'
