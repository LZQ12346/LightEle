from datetime import datetime
import os
import csv

def now():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def remove_csv_headers():
    # 获取当前目录下的所有文件
    files = os.listdir('.')
    for filename in files:
        # 只处理以.csv结尾的文件
        if filename.endswith('.csv'):
            data_rows = []
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # 假设数据行的列数与第一行（表头）的列数相同
                    if row:  # 跳过空行
                        data_rows.append(row)
            if data_rows:
                # 获取表头的列数
                header_length = len(data_rows[0])
                # 过滤掉所有表头行，假设表头行的列数不等于数据行的列数
                data_only = [row for row in data_rows if len(row) == header_length and row != data_rows[0]]
                # 将数据部分写回原文件，覆盖原文件内容
                with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(data_only)
                print(f"已处理文件：{filename}")
            else:
                print(f"文件{filename}为空，已跳过。")