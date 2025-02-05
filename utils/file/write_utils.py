from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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
