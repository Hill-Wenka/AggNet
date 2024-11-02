import os

import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch

from .. import paths
from ..file import check_path, get_basename, is_path_exist

# Maximal solvent-accessible surface area (SASA) of amino acids in a tripeptide (Gly-X-Gly).
# Values are taken from: https://en.wikipedia.org/wiki/Relative_accessible_surface_area
Gly_X_Gly_MaxASA = {
    'ALA': 129.0,
    'ARG': 274.0,
    'ASN': 195.0,
    'ASP': 193.0,
    'CYS': 167.0,
    'GLN': 223.0,
    'GLU': 225.0,
    'GLY': 104.0,
    'HIS': 224.0,
    'ILE': 197.0,
    'LEU': 201.0,
    'LYS': 236.0,
    'MET': 224.0,
    'PHE': 240.0,
    'PRO': 159.0,
    'SER': 155.0,
    'THR': 172.0,
    'TRP': 285.0,
    'TYR': 263.0,
    'VAL': 174.0,
}


def get_atom_array(file_path, clean=True, cache_dir=None, reload=False):
    # 从pdb文件中读取atom_array
    cache_dir = os.path.join(paths.cache, 'atom_array', '') if cache_dir is None else cache_dir
    cache_file = os.path.join(cache_dir, f'{get_basename(file_path)}.npy')
    check_path(cache_file)

    if is_path_exist(cache_file) and not reload:  # 从缓存中读取
        atom_array = torch.load(cache_file, weights_only=False)
    else:  # 从pdb文件中读取并缓存
        atom_array = strucio.load_structure(file_path)
        atom_array = clean_atom_array(atom_array) if clean else atom_array
        torch.save(atom_array, cache_file)
    return atom_array


def write_atom_array(atom_array, file_path):
    # 将atom_array写入pdb文件
    strucio.save_structure(file_path, atom_array)


def write_annotation(atom_array, annotation, file_path):
    # 将annotation数据写入pdb文件的b_factor字段中（用于可视化）
    atom_array.set_annotation('b_factor', annotation)
    write_atom_array(atom_array, file_path)


def clean_atom_array(atom_array):
    # 从atom_array中去除非蛋白质的原子
    return atom_array[atom_array.hetero == False]


def atom_array_to_sequence(atom_array, check=True):
    # 将atom_array转换为氨基酸序列
    try:
        ids, res_names = struc.get_residues(atom_array)
        if check:  # 检查是否为标准氨基酸序列
            illegal_residues = [r for r in res_names if r not in seq.ProteinSequence._dict_3to1.keys()]
            if len(illegal_residues) > 0:
                raise Warning(f"Residues {illegal_residues} are not in the 20 standard amino acids")
        convert_seq = ''.join([seq.ProteinSequence.convert_letter_3to1(r) for r in res_names if r in seq.ProteinSequence._dict_3to1.keys()])
    except Exception as e:
        convert_seq = None
    return convert_seq


def compute_distance_matrix(atom_array, level='residue'):
    if level == 'residue':
        # 计算C-alpha的距离矩阵
        ca = atom_array[atom_array.atom_name == "CA"]  # Filter only CA atoms
        distance_matrix = np.array([struc.distance(c, ca) for c in ca])
    elif level == 'atom':
        raise NotImplementedError
    else:
        raise RuntimeError(f'No such pre-defined level: {level}')
    return distance_matrix


def compute_adjacency_matrix(atom_array, threshold=8, level='residue'):
    if level == 'residue':
        # 根据距离阈值计算C-alpha的邻接矩阵
        ca = atom_array[atom_array.atom_name == "CA"]  # Filter only CA atoms
        cell_list = struc.CellList(ca, cell_size=threshold)  # Create cell list of the CA atom array for efficient computation
        adjacency_matrix = cell_list.create_adjacency_matrix(threshold)  # default threshold is 8 Angstrom
    elif level == 'atom':
        raise NotImplementedError
    else:
        raise RuntimeError(f'No such pre-defined level: {level}')
    return adjacency_matrix


def construct_graph(atom_array, threshold=8, level='residue'):
    # 构建蛋白质的图结构
    if level == 'residue':
        nodes = np.array(list(atom_array_to_sequence(atom_array)))  # (L,)
        edges = compute_adjacency_matrix(atom_array, threshold).astype(int)  # (L, L)
        distances = compute_distance_matrix(atom_array)  # (L, L)
    elif level == 'atom':
        raise NotImplementedError
    else:
        raise RuntimeError(f'No such pre-defined level: {level}')
    return nodes, edges, distances


def get_neighbors(atom_coord, atom_array, cell_list=None, cell_size=5, radius=8.0, near_residues=True):
    # 获取给定坐标atom_coord附近的原子/残基信息
    # atom_coord: (N, 3), e.g. np.array([[x1, y1, z1], [x2, y2, z2], ...])
    atom_coord = np.array(atom_coord) if isinstance(atom_coord, list) else atom_coord
    cell_list = struc.CellList(atom_array, cell_size=cell_size) if cell_list is None else cell_list
    atom_indices = cell_list.get_atoms(atom_coord, radius=radius)
    near_atoms = atom_array[atom_indices]

    if near_residues:
        residue_indices, near_residue_names = struc.get_residues(near_atoms)
        residue_indices = set(residue_indices)
        near_residue_atoms = None
        for x in [atom_array[atom_array.res_id == res_id] for res_id in residue_indices]:
            if near_residue_atoms is None:
                near_residue_atoms = x
            else:
                near_residue_atoms = near_residue_atoms + x
        near_residue_c_alpha = near_residue_atoms[near_residue_atoms.atom_name == "CA"]
        neighbors = {'atom_indices': atom_indices,
                     'near_atoms': near_atoms,
                     'residue_indices': residue_indices,
                     'near_residue_atoms': near_residue_atoms,
                     'near_residue_c_alpha': near_residue_c_alpha}
    else:
        neighbors = {'atom_indices': atom_indices, 'near_atoms': near_atoms}
    return neighbors


def compute_sasa(structure, level='residue', mode='SASA', **params):
    # 计算蛋白质的SASA/RSA，包括原子级和残基级
    if type(structure) == str:
        atom_array = strucio.load_structure(structure)
    else:
        atom_array = structure

    atom_sasa = struc.sasa(atom_array, **params)  # compute atom-level SASA
    atom_sasa = np.nan_to_num(atom_sasa) # H元素的SASA为nan，需要转换为0
    if level == 'residue':
        res_sasa = struc.apply_residue_wise(atom_array, atom_sasa, np.sum)  # compute residue-level SASA
        if mode == 'SASA':
            return res_sasa
        elif mode == 'RSA':
            res_ids, res_names = struc.get_residues(atom_array)  # achieve residue list
            maxASA = np.array([Gly_X_Gly_MaxASA[r] for r in res_names])  # compute MaxASA
            res_rsa = res_sasa / maxASA
            return res_rsa
        else:
            raise RuntimeError(f'No such pre-defined mode: {mode}')
    elif level == 'atom':
        if mode == 'SASA':
            return atom_sasa
        elif mode == 'RSA':
            raise RuntimeError('Atom level not support RSA!')
        else:
            raise RuntimeError(f'No such pre-defined mode: {mode}')
    else:
        raise RuntimeError(f'No such pre-defined level: {level}')
