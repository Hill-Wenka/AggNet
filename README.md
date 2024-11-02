# AggNet

## Clone and Create environment

Clone the git repository and then create the conda environment as follows.

```
# Install the required packages
conda create -n AggNet python=3.11 -y
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nb_conda_kernels jupyter ipywidgets openpyxl pandas matplotlib seaborn scikit-learn biopython biotite -c conda-forge
pip install jupyterlab python-Levenshtein easydict tqdm numpy scipy tensorboard omegaconf fair-esm h5py aaindex einops lightning==2.4.0
```

## Download the model checkpoint

Download the model checkpoint from the following link and place it in the `./checkpoint` directory.
Checkpoint: https://drive.google.com/file/d/1inplkuo_EqtO-HwAs-UXEIhExxDE6uKt/view?usp=sharing

## Predict amyloid propensity of peptides
```
# use Hex142 dataset as an example
conda actvate AggNet
python ./script/predict_amyloid.py --fasta ./data/AmyHex/Hex142.fasta --batch_size 256 --checkpoint ./checkpoint/APNet.ckpt --output ./APNet_results.csv
```

## Profile a protein sequence
```
# use WFL VH as an example
conda actvate AggNet
python ./script/predict_APR.py --sequence QVQLVQSGAEVKKPGSSVKVSCKASGGTFWFGAFTWVRQAPGQGLEWMGGIIPIFGLTNLAQNFQGRVTITADESTSTVYMELSSLRSEDTAVYYCARSSRIYDLNPSLTAYYDMDVWGQGTMVTVSS --checkpoint ./checkpoint/APNet.ckpt --output ./APRNet_results.csv
```