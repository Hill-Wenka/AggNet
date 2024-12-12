# add current dir to PYTHONPATH
import sys

sys.path.append('.')

import argparse
import pandas as pd
from model.APRNet import APRNet
from model.APNet.data_module import DataModule
from model.APNet.lightning_module import LightningModule
from utils.lightning import LitModelInference

parser = argparse.ArgumentParser(description='Analyze Aggregation Profile or Identify APR of Proteins using APRNet')
parser.add_argument('--sequence', type=str, default='QVQLVQSGAEVKKPGSSVKVSCKASGGTFWFGAFTWVRQAPGQGLEWMGGIIPIFGLTNLAQNFQGRVTITADESTSTVYMELSSLRSEDTAVYYCARSSRIYDLNPSLTAYYDMDVWGQGTMVTVSS',
                    help='Protein sequence to be profiled, default is WFL VH')
parser.add_argument('--structure', type=str, default=None,
                    help='Path to PDB file of the protein structure')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/APNet.ckpt',
                    help='Path to model checkpoint')
parser.add_argument('--output', type=str, default='./APRNet_results.csv',
                    help='Path to save prediction results')

APRNet_struct_params = {
    'beta': 3.36,
    'delta': 0.4,
    't_start': 0.51,
    't_expand': 0.37,
    't_patience': 9,
}
APRNet_seq_params = {
    't_start': 0.46,
    't_expand': 0.37,
    't_patience': 7,
}

if __name__ == '__main__':
    args = parser.parse_args()
    sequence = args.sequence
    structure = args.structure
    checkpoint = args.checkpoint
    output = args.output
    params = APRNet_struct_params if structure is not None else APRNet_seq_params
    structure = None if structure is None else [structure]

    # load model
    APNet = LitModelInference(LightningModule, DataModule, checkpoint)
    aprnet = APRNet.APRNet(APNet)

    # inference
    pred_labels, pred_scores = aprnet([sequence], structure, **params)
    labels, scores = pred_labels[0], pred_scores[0]

    df = pd.DataFrame({'residue': list(sequence),
                       'scores': scores,
                       'APR': labels})
    df.to_csv(output, index=False)
    print(f'\nResults saved to {output}\n')
    print(df)
