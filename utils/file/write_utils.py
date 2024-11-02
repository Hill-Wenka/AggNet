import json

import numpy as np
import pandas as pd
import torch
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def data2file(data, path, **kwargs):
    # 将数据保存至硬盘，根据输出路径后缀判断输出文件类型。写入成功则返回True，否则会报错
    suffix = path.split(".")[-1]
    if suffix == "fasta":
        write_fasta(path, data, **kwargs)
    elif suffix == "pt":
        torch.save(data, path, **kwargs)
    elif suffix == "npy":
        np.save(path, data, **kwargs)
    elif suffix == "xlsx":
        data.to_excel(path, **kwargs)
    elif suffix == "tsv":
        data = pd.DataFrame(data) if type(data) != pd.DataFrame else data
        data.to_csv(path, sep="\t", **kwargs)
    elif suffix == "csv":
        data = pd.DataFrame(data) if type(data) != pd.DataFrame else data
        data.to_csv(path, sep=",", **kwargs)
    elif suffix == "yaml":
        with open(path, "w") as f:
            yaml.dump(data, f, **kwargs)
    elif suffix == "json":
        with open(path, "w") as f:
            json.dump(data, f, **kwargs)
    else:
        write_file(data, path, **kwargs)
    return True


def write_file(text, file, **kwargs):
    # 将文本写入文件
    with open(file, "w") as f:
        f.write(text)


def write_fasta(path, seqs, custom_index=None, description=None):
    """
    调取biopython包输出fasta文件
    :param path: 输出的目标路径
    :param seqs: 序列列表
    :param custom_index: 自定义索引，如果为None则使用默认index，从0至len(seqs)-1
    :param description: 序列描述 或者 标签列表
    """
    custom_index = [str(i) for i in range(len(seqs))] if custom_index is None else custom_index
    records = []
    for i in range(len(seqs)):
        if description is None:
            seq_record = SeqRecord(Seq(seqs[i]), id=custom_index[i], description="")
        else:
            seq_record = SeqRecord(Seq(seqs[i]), id=custom_index[i], description=f"| {description[i]}")
        records.append(seq_record)
    try:
        SeqIO.write(records, path, "fasta")
    except Exception:
        raise RuntimeError("Failed to write fasta")


def write_yaml(path, data, **kwargs):
    # 将数据写入yaml文件
    assert ".yaml" in path, "output file must be a yaml file"
    with open(path, "w") as f:
        yaml.dump(data, f, **kwargs)


def write_data_label_pair_file(path, seqs, labels, custom_index=None):
    """
    将seq_list, label_list输出至xlsx, csv, tsv等类表格格式文件，写入成功则返回True，否则会报错
    :param path: 输出的目标路径
    :param seqs: 序列列表
    :param labels: 标签列表
    :param custom_index: 自定义索引，如果为None则使用默认index，从0至len(seqs)-1
    :return: True
    """
    custom_index = [i for i in range(len(seqs))] if custom_index is None else custom_index
    df = pd.DataFrame({"Index": custom_index, "Data": seqs, "Label": labels})
    if "xlsx" in path:
        df.to_excel(path, index=False)
    else:
        sep = "\t" if ".tsv" in path else ","
        df.to_csv(path, sep=sep, index=False)
    return True
