import os
import pandas as pd
from werkzeug.datastructures import FileStorage
import zipfile
from .tools import now


def check_nk_csv_valid(file_path):
    """
    Check if the refractive index data is valid. Follow the following rules:
    1. Must be 3 columns: Wavelength (nm), n, k. No headers are ok.
    2. At least 2 rows.
    3. n and k must be numbers.
    :param file_path: FileStorage object
    :return:
    """
    df = pd.read_csv(file_path)
    if df.shape[1] != 3 or df.shape[0] < 2:  # Must be 3 columns
        return False
    return True


def find_nk_file_by_material(material: str):
    """
    :param material:
    :return: None or str
    """
    nk_dir = os.path.join("refractive_index_csv")
    for dirpath, dirnames, filenames in os.walk(nk_dir):
        for csv_file in filenames:
            if f"{material.lower()}.csv" == csv_file:
                return os.path.join(dirpath, csv_file)
    return None


def zip_files(source_dir, output_zip_dir, filename):
    """
    将指定目录下的所有文件打包成一个zip文件，并放到指定路径下。

    :param source_dir: 需要打包的目录路径
    :param output_zip_dir: 生成的zip文件的保存路径
    :param filename: 生成的zip文件的文件名
    """
    if not os.path.exists(output_zip_dir):
        os.makedirs(output_zip_dir)
    output_zip_path = str(os.path.join(output_zip_dir, filename))
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目录中的所有文件和文件夹
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # 计算文件的完整路径
                file_path = os.path.join(root, file)
                # 将文件添加到zip文件中，同时去掉文件的目录部分（以确保zip文件中没有目录结构）
                zipf.write(file_path, os.path.relpath(file_path, source_dir))
