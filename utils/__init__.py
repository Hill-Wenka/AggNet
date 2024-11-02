import os
import sys

from omegaconf import OmegaConf

'''
Set the PYTHON_PATH and store constants
'''
# project root path: '/home/hew/python/AggNet/'
utils_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(utils_path)[0]

if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(utils_path)
    print('==================== add root_path to sys.path ====================')
    print('root_path:', root_path)
    print('======================================================================================')

paths = OmegaConf.create()
paths['root'] = os.path.join(root_path, '')
paths['data'] = os.path.join(root_path, 'data', '')
paths['dataset'] = os.path.join(root_path, 'dataset', '')
paths['result'] = os.path.join(root_path, 'result', '')
paths['script'] = os.path.join(root_path, 'script', '')
paths['temp'] = os.path.join(root_path, 'temp', '')
paths['utils'] = os.path.join(utils_path, '')
paths['cache'] = os.path.join(root_path, 'cache', '')

# modify the following paths according to your own environment
conda_path = '/home/hew/anaconda3/'
conda_env = 'AggNet'
