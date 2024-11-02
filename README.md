# AggNet

## Installation and running

### Clone and Create environment

Clone the git repository and then create the conda environment as follows.

```commandline
conda create -n GPTDesign python=3.12 -y
conda activate GPTDesign
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install matplotlib ipywidgets jupyterlab python-Levenshtein scikit-learn numba easydict aaindex tqdm numpy scipy pandas seaborn tensorboard biopython h5py loguru omegaconf openpyxl einops redis biotite fair-esm lightning torch_geometric 
```
