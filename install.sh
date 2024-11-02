conda create -n prot-gen-env python=3.11 -y
#conda activate prot-gen-env # run in command line
#souce {ANACONDA_PATH}/bin/activate AggNet # run in script, ANACONDA_PATH is your path of anaconda3
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nb_conda_kernels jupyter ipywidgets openpyxl pandas matplotlib seaborn scikit-learn biopython biotite -c conda-forge
pip install jupyterlab python-Levenshtein easydict tqdm numpy scipy tensorboard omegaconf fair-esm h5py aaindex einops lightning==2.4.0
