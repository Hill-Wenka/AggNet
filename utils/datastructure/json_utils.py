import json

import omegaconf


def json2list(param):
    # 根据param的数据类型进行相应的转换，最终都转换为list
    if isinstance(param, str):
        param = json.loads(param)
    elif isinstance(param, int):
        param = [param]
    elif isinstance(param, (list, omegaconf.listconfig.ListConfig)):
        pass
    else:
        raise RuntimeError(f'Parameter has illegal type: [{type(param)}]')
    return param
