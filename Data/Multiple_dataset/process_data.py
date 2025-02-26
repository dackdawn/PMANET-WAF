import csv
from collections import defaultdict

def process_csv_to_txt(input_csv, benign_output_txt, malware_output_txt):
    # 原有的二分类处理函数保持不变
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        benign_urls = []
        malware_urls = []
        
        for row in reader:
            url, label = row
            if label == 'benign':
                benign_urls.append(url)
            else:
                malware_urls.append(url)
    
    with open(benign_output_txt, 'w', encoding='utf-8') as benign_file:
        for url in benign_urls:
            benign_file.write(url + '\n')
    
    with open(malware_output_txt, 'w', encoding='utf-8') as malware_file:
        for url in malware_urls:
            malware_file.write(url + '\n')

def process_csv_to_multiclass_txt(input_csv):
    """
    将CSV文件按照不同的标签分别生成对应的txt文件
    每个标签生成一个{label}_urls.txt文件
    """
    # 使用defaultdict收集不同标签的URL
    label_urls = defaultdict(list)
    
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        
        for row in reader:
            url, label = row
            label_urls[label].append(url)
    
    # 为每个标签创建对应的文件
    for label, urls in label_urls.items():
        output_file = f'{label}_urls.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for url in urls:
                f.write(url + '\n')
    
    return list(label_urls.keys())  # 返回所有标签列表

if __name__ == '__main__':
    input_csv = 'kaggle_multi.csv'

    # 生成二分类数据集
    benign_output_txt = 'benign_urls.txt'
    malware_output_txt = 'malware_urls.txt'
    process_csv_to_txt(input_csv, benign_output_txt, malware_output_txt)
    print("完成生成二分类数据集")

    # 生成多分类数据集
    labels = process_csv_to_multiclass_txt(input_csv)
    print("完成生成多分类数据集")
    print(f"生成的标签类别: {labels}")
