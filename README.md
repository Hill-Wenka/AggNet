# AggNet

## Clone and Create environment

Clone the git repository and then create the conda environment as follows.

```
# Install the required packages
conda create -n AggNet python=3.11 -y
conda activate AggNet
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nb_conda_kernels jupyter ipywidgets openpyxl pandas matplotlib seaborn scikit-learn biopython biotite -c conda-forge
pip install jupyterlab python-Levenshtein easydict tqdm numpy scipy tensorboard omegaconf fair-esm h5py aaindex einops lightning==2.4.0
```

## Download the model checkpoint

Download the model checkpoint from the following link and place it in the `./checkpoint` directory.
Checkpoint: https://drive.google.com/file/d/1inplkuo_EqtO-HwAs-UXEIhExxDE6uKt/view?usp=sharing

## Predict amyloid propensity of peptides
Use Hex142 dataset as an example
```
conda actvate AggNet
python ./script/predict_amyloid.py --fasta ./data/AmyHex/Hex142.fasta --batch_size 256 --checkpoint ./checkpoint/APNet.ckpt --output ./APNet_results.csv
```
Use CPAD2.0 dataset as an example
```
conda actvate AggNet
python ./script/predict_amyloid.py --fasta ./data/CPAD2/CPAD2.fasta --batch_size 256 --checkpoint ./checkpoint/APNet.ckpt --output ./APNet_CPAD2_results.csv
```
For usage information, run
```
python ./script/predict_amyloid.py -h

--------------------------------------------------------------------------------

Predict Amyloidogenic Peptides using APNet

options:
  -h, --help            show this help message and exit
  --fasta FASTA         Path to input fasta file
  --batch_size BATCH_SIZE
                        Batch size for prediction
  --checkpoint CHECKPOINT
                        Path to model checkpoint
  --output OUTPUT       Path to save prediction results
```

## Profile a protein sequence
Use WFL VH as an example
```
conda actvate AggNet
python ./script/predict_APR.py --sequence QVQLVQSGAEVKKPGSSVKVSCKASGGTFWFGAFTWVRQAPGQGLEWMGGIIPIFGLTNLAQNFQGRVTITADESTSTVYMELSSLRSEDTAVYYCARSSRIYDLNPSLTAYYDMDVWGQGTMVTVSS --checkpoint ./checkpoint/APNet.ckpt --output ./APRNet_results.csv
```
For usage information, run
```
python ./script/predict_amyloid.py -h

--------------------------------------------------------------------------------

Analyze Aggregation Profile or Identify APR of Proteins using APRNet

options:
  -h, --help            show this help message and exit
  --sequence SEQUENCE   Protein sequence to be profiled, default is WFL VH
  --structure STRUCTURE
                        Path to PDB file of the protein structure
  --checkpoint CHECKPOINT
                        Path to model checkpoint
  --output OUTPUT       Path to save prediction results
```

## Easy Start With Jupyter Notebook
1. see `./example.ipynb` for a quick start with Jupyter Notebook.
2. see `./CPAD2.ipynb` for the analysis of predictions on CPAD2.0 dataset.
