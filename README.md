# AggNet

## Clone and Create environment

Clone the git repository and then create the conda environment as follows.

```Install the required packages
conda create -n prot-gen-env python=3.11 -y
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nb_conda_kernels jupyter ipywidgets openpyxl pandas matplotlib seaborn scikit-learn biopython biotite -c conda-forge
pip install jupyterlab python-Levenshtein easydict tqdm numpy scipy tensorboard omegaconf fair-esm h5py aaindex einops lightning==2.4.0
```

## Download the model checkpoint

Download the model checkpoint from the following link and place it in the `./checkpoint` directory.
Checkpoint: https://drive.google.com/file/d/1inplkuo_EqtO-HwAs-UXEIhExxDE6uKt/view?usp=sharing