import Levenshtein
import numpy as np
from tqdm.notebook import tqdm
# from tqdm import tqdm

from ..parallel import concurrent_submit


def distance(s1, s2, dist='Levenshtein'):
    # 计算两个序列之间的距离
    if dist == 'Levenshtein':
        d = Levenshtein.distance(s1, s2)
    elif dist == 'Hamming':
        d = sum([bool(a != b) for a, b in zip(s1, s2)])
    else:
        raise RuntimeError(f'No such pre-defined dist: {dist}')
    return d


def parallel_compute(i, j, seq1, seq2, dist):
    d_ij = distance(seq1, seq2, dist=dist)
    return i, j, d_ij


def distance_matrix(seqs, only_upper=True, dist='Levenshtein', parallel=False):
    # 输入序列列表，返回所有序列对的距离矩阵
    seqs = list(seqs)
    matrix = np.zeros([len(seqs), len(seqs)])

    if parallel:
        d_list = concurrent_submit(parallel_compute, [(i, j, seqs[i], seqs[j], dist) for i in range(len(seqs)) for j in range(len(seqs)) if j > i])
        for i, j, d_ij in d_list:
            matrix[i][j] = d_ij
            if not only_upper:
                matrix[j][i] = d_ij
    else:
        for i, pep1 in tqdm(enumerate(seqs), total=len(seqs)):
            for j, pep2 in enumerate(seqs):
                if j > i:
                    d_ij = distance(pep1, pep2, dist=dist)
                    matrix[i][j] = d_ij
                    if not only_upper:
                        matrix[j][i] = d_ij
    return matrix


def mutate(sequence, mutation_string=None, seperator='/', chains=None, positions=None, mutations=None, zero_based=False):
    # sequence: wild-type sequence
    if isinstance(sequence, str):  # 单链。默认是A链
        sequence = {'A': list(sequence)}
    elif isinstance(sequence, dict):  # 多链。key是chain_id，value是序列
        sequence = {k: list(v) for k, v in sequence.items()}
    else:
        raise RuntimeError(f'No such pre-defined sequence type: {type(sequence)}')

    if mutation_string is not None:  # mutation_string: 'A1T A2C A3G' or 'A1T,A2C,A3G'
        assert positions is None and mutations is None
        mutation_list = mutation_string.split(seperator)
        # print('mutation_list', mutation_list)
        wt_residues, chain_positions, mt_residues = zip(*[(x[0], x[1:-1], x[-1]) for x in mutation_list])
        chains = [x[0] if x[0].isalpha() else 'A' for x in chain_positions]  # isalpha()判断是否是字母，若是则指定了chain_id，否则默认是A链
        positions = [int(x[1:]) if x[0].isalpha() else int(x) for x in chain_positions]
        positions = [x - 1 for x in
                     positions] if not zero_based else positions  # zero_based: 位置索引是否是从0开始。若不是，则默认从1开始，需要先减1
    else:
        assert positions is not None and mutations is not None
        chains = ['A'] * len(positions) if chains is None else chains
        positions = [x - 1 for x in positions] if not zero_based else positions
        wt_residues = [sequence[chain][position] for chain, position in zip(chains, positions)]
        mt_residues = mutations

    for chain, wt_res, position, mt_res in zip(chains, wt_residues, positions, mt_residues):
        # print(wt_res, chain, position, mt_res)
        assert sequence[chain][position] == wt_res, \
            f'{chain}{position} is {sequence[chain][position]} not {wt_res}'  # 检查wild-type residue是否正确
        sequence[chain][position] = mt_res

    # print('sequence', sequence)
    sequence = {chain: ''.join(seq) for chain, seq in sequence.items()} \
        if len(sequence) > 1 else ''.join(list(sequence.values())[0])
    # 如果是单链，则返回突变序列；如果是多链，则返回字典，key是chain_id，value是突变后的序列
    return sequence


def format_mutation(wt_seq, mt_seq, chain_id='A', offset=0, seperator='/', end_seperator='', omit_chain=True):
    # 根据wild-type和mutant序列，生成mutation_string
    assert len(wt_seq) == len(mt_seq)
    mut_list = ['{}{}{}{}'.format(wt_seq[i], '' if omit_chain else chain_id, i + 1 + offset, mt_seq[i])
                for i in range(len(wt_seq)) if wt_seq[i] != mt_seq[i]]
    mutation_string = seperator.join(mut_list) + end_seperator
    return mutation_string
