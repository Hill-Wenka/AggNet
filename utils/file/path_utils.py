import glob
import os
import shutil


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


def makedir(path):
    os.makedirs(path, exist_ok=True)  # 递归创建文件夹，如果文件夹已经存在则不会报错


def walk_path(base):
    # 遍历base文件夹（目录），返回所有的路径组合（root, dir, file）
    return [[root, dirs, files] for root, dirs, files in os.walk(base)]


def list_dir(base, absolute=False):
    # 遍历base文件夹（目录），返回当前文件夹下的所有子文件夹
    if absolute:  # 返回绝对路径
        return [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    else:
        return [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]


def list_file(base, absolute=False):
    # 遍历base文件夹（目录），返回当前文件夹下的所有子文件
    if absolute:  # 返回绝对路径
        return [os.path.join(base, f) for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
    else:
        return [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]


def filter_dir(path, pattern, absolute=False, recursive=False):
    # 递归遍历指定文件夹下的所有文件夹，匹配符合指定模式的文件夹
    dirs = glob.glob(pattern, root_dir=path, recursive=recursive)
    dirs = [dir for dir in dirs if os.path.isdir(os.path.join(path, dir))]
    dirs = [os.path.join(path, dir) for dir in dirs] if absolute else dirs
    return dirs


def filter_file(path, pattern, absolute=False, recursive=False):
    # 递归遍历指定文件夹下的所有文件，匹配符合指定模式的文件
    files = glob.glob(pattern, root_dir=path, recursive=recursive)
    files = [os.path.abspath(os.path.join(path, file)) for file in files] if absolute else files
    return files


def remove_dir(dirs, force=True):
    # 删除指定的文件夹(列表)
    if isinstance(dirs, str):
        dirs = [dirs]
    for dir in dirs:
        if os.path.isdir(dir):
            if force:
                os.system(f'rm -rf {dir}')
            else:
                os.rmdir(dir)
        else:
            print(f'Error: {dir} is not a directory')


def remove_file(files):
    # 删除指定的文件(列表)
    if isinstance(files, str):
        files = [files]
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
        else:
            print(f'Warning: {file} is not a file')


def rename_file(file, new_name):
    # 批量重命名文件
    if isinstance(file, str):
        file = [file]
    if isinstance(new_name, str):
        new_name = [new_name]
    assert len(file) == len(new_name)
    for f, n in zip(file, new_name):
        if os.path.isfile(f):
            os.rename(f, n)
        else:
            print(f'Error: {f} is not a file')


def copy_file(src_path, target_path):
    # 复制文件到指定路径
    if is_path_exist(src_path):
        check_path(target_path)
        shutil.copy(src_path, target_path)
        print(f'Copy {src_path} to {target_path}: Success')
    else:
        print(f'Error: {src_path} is not a file')


def get_basename(path, suffix=False):
    # 获取文件名
    return os.path.basename(path) if suffix else os.path.splitext(os.path.basename(path))[0]
