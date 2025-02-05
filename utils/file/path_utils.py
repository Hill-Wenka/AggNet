import os


def is_path_exist(path):
    # 检查路径是否存在
    return os.path.exists(path)


def check_path(path, mkdir=True, log=True):
    # 检查路径所在文件夹是否存在, 如果路径不存在则自动新建
    dir = path if os.path.isdir(path) else os.path.abspath(os.path.dirname(path))  # 如果path是文件夹则直接使用path，否则使用path的父目录
    is_exist = is_path_exist(dir)
    if mkdir and not is_path_exist(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            if log:
                print(f'The path does not exist, makedir: {dir}: Success')
        except Exception:
            raise RuntimeError(f'The path does not exist, makedir {dir}: Failed')
    return is_exist


def list_file(base, absolute=False):
    # 遍历base文件夹（目录），返回当前文件夹下的所有子文件
    if absolute:  # 返回绝对路径
        return [os.path.join(base, f) for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
    else:
        return [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
