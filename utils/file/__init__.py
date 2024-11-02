import zipfile

from .path_utils import *
from .read_utils import *
from .write_utils import *


def zip_files(output, files):
    # 将指定的文件打包成zip文件，仅支持文件，不支持文件夹
    assert '.zip' in output, 'output file must be a zip file'
    with zipfile.ZipFile(output, 'w') as zipf:
        for file in files:
            zipf.write(file, arcname=os.path.basename(file))
