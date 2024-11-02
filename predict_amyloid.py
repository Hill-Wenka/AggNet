# add current dir to PYTHONPATH
import sys

sys.path.append('.')

import argparse
import pandas as pd
from model.APNet.data_module import DataModule
from model.APNet.lightning_module import LightningModule
from utils.lightning import LitModelInference, merge_batch_prediction
from utils.file import read_fasta

parser = argparse.ArgumentParser(description='Predict amyloidogenic peptides using APNet')
parser.add_argument('--fasta', type=str, default='./data/AmyHex/Hex142.fasta',
                    help='Path to input fasta file')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for prediction')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/APNet.ckpt',
                    help='Path to model checkpoint')
parser.add_argument('--output', type=str, default='./APNet_results.csv',
                    help='Path to save prediction results')

if __name__ == '__main__':
    args = parser.parse_args()
    fasta_file = args.fasta
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    output = args.output

    # load data
    peptides, _ = read_fasta(fasta_file)

    # load model
    APNet = LitModelInference(LightningModule, DataModule, checkpoint)
    APNet.set_batch_size(batch_size=batch_size, num_workers=1)

    # inference
    predictions = APNet.predict(peptides)
    results = merge_batch_prediction(predictions)
    probs = results['preds'].cpu().softmax(dim=-1).numpy()
    pred_labels = ['amyloid' if p[1] > 0.5 else 'non-amyloid' for p in probs]
    df = pd.DataFrame({'peptide': peptides,
                       'probability': probs[:, 1],
                       'label': pred_labels})
    df.to_csv(output, index=False)
    print(f'\nResults saved to {output}\n')
    print(df)
