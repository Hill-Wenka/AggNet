import json
import string
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from Bio import SeqIO


def file2data(path, **kwargs):
    # 统一化文件读取。从硬盘读取文件，根据文件路径后缀判断读取文件的类型，返回文件中的数据
    suffix = path.split(".")[-1]
    if suffix == "fasta":
        data = read_fasta(path, **kwargs)
    elif suffix == "pt":
        data = torch.load(path, weights_only=False, **kwargs)
    elif suffix == "npy":
        data = np.load(path, **kwargs)
    elif suffix == "xlsx":
        data = pd.read_excel(path, **kwargs)
    elif suffix == "tsv":
        sep = kwargs.pop("sep", "\t")
        data = pd.read_csv(path, sep=sep, **kwargs)
    elif suffix == "csv":
        data = pd.read_csv(path, **kwargs)
    elif suffix == "yaml":
        data = read_yaml(path, **kwargs)
    elif suffix == "json":
        data = read_json(path, **kwargs)
    else:
        data = read_file(path)  # 读取文本文件
    return data


def read_file(path):
    # 读取文本文件，返回文件中的文本内容
    with open(path, "r") as f:
        text = f.read()
    return text


def read_fasta(path, **kwargs):
    # 调取biopython包读取fasta文件，返回fasta中每一条记录的序列以及对应的描述
    seqs = [str(fa.seq) for fa in SeqIO.parse(path, "fasta")]
    description = [fa.description for fa in SeqIO.parse(path, "fasta")]
    return seqs, description


def read_yaml(path, encoding="utf-8"):
    # 读取yaml并转为dict，主要用于从yaml文件读取默认的项目配置以及模型超参数
    try:
        with open(path, encoding=encoding) as file:
            yaml_dict = yaml.load(file.read(), Loader=yaml.FullLoader)
    except Exception:
        raise RuntimeError(
            "Failed to read the yaml file, the specific encoding is wrong"
        )
    return yaml_dict


def read_json(path, **kwargs):
    # 读取json文件，返回对应的数据字典
    with open(path, "r") as load_f:
        data = json.load(load_f, **kwargs)
    return data


def read_seq_label_pair_file(path, index_col=None, data_col=None, label_col=None):
    # 专门处理类似seq-label pair风格的数据，从xlsx, csv, tsv等类表格格式文件中读取数据并转换为 data, labels 列表
    with open(path, "r") as file:
        for line in file:
            if line[-1] == "\n":
                line = line[:-1]
            columns = line.split("\t")
            break

    index_cols = ["index", "Index", "idx", "Idx"]
    if index_col is None:
        for col in index_cols:
            if col in columns:
                index_col = col
                break
    if index_col is None:  # 如果没有找到预定义的index_col，则抛出异常
        raise RuntimeError(
            'Please specify param "index_col" since no default keywords match'
        )

    data_cols = ["sequence", "Sequence", "seq", "Seq", "data", "Data"]
    if data_col is None:
        for col in data_cols:
            if col in columns:
                data_col = col
                break
    if data_col is None:  # 如果没有找到预定义的data_col，则抛出异常
        raise RuntimeError(
            'Please specify param "data_col" since no default keywords match'
        )

    label_cols = ["label", "Label", "class", "Class"]
    if label_col is None:
        for col in label_cols:
            if col in columns:
                label_col = col
                break
    if label_col is None:  # 如果没有找到预定义的label_col，则抛出异常
        raise RuntimeError(
            'Please specify param "label_col" since no default keywords match'
        )

    if "xlsx" not in path:
        df = pd.read_excel(path, index_col=index_col)
    else:
        sep = "\t" if ".tsv" in path else ","
        df = pd.read_csv(path, sep=sep, index_col=index_col)

    data = df[data_col].tolist()
    labels = df[label_col].tolist()
    return data, labels


'''
A2M/A3M is a family of formats derived from FASTA used for sequence alignments. 
In A2M/A3M sequences, lowercase letters are taken to mean insertions, and are shown as dot (".") characters in other sequences. 
Dots can be discarded for compactness without losing information. 
As with typical FASTA used in alignments, gaps ("-") are taken to mean exactly one position. 
A3M is like A2M, with the added rule that gaps aligned with insertions can also be discarded.
'''
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str, keep_raw=False) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    # .a2m and .a3m are the same format, but .a3m can have lower case letters
    # keep_raw is used to keep the original sequences, without removing insertions
    if keep_raw:
        msa = [(record.description, str(record.seq)) for record in SeqIO.parse(filename, "fasta")]
    else:
        msa = [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]
    return msa
