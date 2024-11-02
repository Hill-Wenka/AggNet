import argparse
import json
import os
from argparse import Namespace

import omegaconf
from omegaconf import OmegaConf

from .. import root_path


# from utils import root_path


def log_config(config, prefix='config'):
    # 格式化输出配置参数config (dict/Namespace/omegaconf.dictconfig.DictConfig)
    if type(config) == dict:
        pass
    elif type(config) == omegaconf.dictconfig.DictConfig:
        config = OmegaConf.to_object(config)
    elif type(config) == argparse.Namespace:
        config = vars(config)
    else:
        raise RuntimeError(f'Param args is with illegal type: {config}')
    print(f'[{prefix}]:')
    print(json.dumps(config, indent=4, ensure_ascii=False))  # 缩进4空格，中文字符不转义成Unicode


def config_to_namespace(config):
    # 将config从OmegaConf转换为Namespace
    return Namespace(**OmegaConf.to_object(config))


def namespace_to_config(config):
    # 将config从Namespace转换为OmegaConf
    return OmegaConf.create(vars(config))


def config_to_dict(config):
    # 将config从OmegaConf转换为dict
    return OmegaConf.to_object(config)


def dict_to_config(config_dict):
    # 将config从dict转换为OmegaConf
    return OmegaConf.create(config_dict)


def load_config(path):
    # 从checkpoints的hparams文件中加载保存好的参数设置
    config = OmegaConf.load(path)
    try:  # lightning保存的参数设置默认是在args字段中
        config = config.config
    except:  # 读取的是一般的yaml文件，没有args字段
        pass
    return config


def merge_config(config, *args):
    # 更新config, 统一处理dict类型，yaml文件路径和命令行参数
    for x in args:
        if type(x) == dict:
            # dict: key-value pairs
            config = OmegaConf.merge(config, x)
        elif type(x) == str:
            if x.endswith('.yaml') and '=' not in x:
                # yaml path
                config = OmegaConf.merge(config, OmegaConf.load(x))
            else:
                # command line arguments
                config = OmegaConf.merge(config, OmegaConf.from_dotlist([x]))
        else:
            raise RuntimeError(f'Invalid type: {type(x)} of {x}')
    return config


def parse_config(*args):
    '''
    加载配置，包括项目设置和模型超参数。考虑从yaml文件中读取，从自定义参数字典中读取，以及从命令行参数中读取。
    :param *args: 更新的参数设置
    :return: args: OmegaConf (omegaconf.dictconfig.DictConfig)
    '''
    cmd_args = os.sys.argv[1:]

    if len(cmd_args) > 0:
        if cmd_args[0] == '-f':
            cmd_args = cmd_args[2:]
        print('cmd_args', cmd_args)

    default_config = OmegaConf.create()
    default_config_list = ['utils/config/default_config/project/project.yaml',
                           # 'utils/config/default_config/project/nni.yaml',
                           'utils/config/default_config/lightning/LitData.yaml',
                           'utils/config/default_config/lightning/LitModel.yaml',
                           'utils/config/default_config/dataset/MyDataset.yaml',
                           'utils/config/default_config/model/MyModel.yaml']
    default_config_list = [os.path.join(root_path, x) for x in default_config_list]
    default_config = merge_config(default_config, *default_config_list)
    config = merge_config(default_config, *args, *cmd_args)
    return config


if __name__ == '__main__':
    config = parse_config()
    print('========== format config ==========')
    log_config(config)
