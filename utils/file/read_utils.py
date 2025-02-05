from Bio import SeqIO


def read_fasta(path, **kwargs):
    # 调取biopython包读取fasta文件，返回fasta中每一条记录的序列以及对应的描述
    seqs = [str(fa.seq) for fa in SeqIO.parse(path, "fasta")]
    description = [fa.description for fa in SeqIO.parse(path, "fasta")]
    return seqs, description
