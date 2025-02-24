import csv

"""
本文件用于将下面这种的csv文件转为txt
源文件（第一行为标题行，第一列为url，第二列为label）：
url,label
br-icloud.com.br,phishing
mp3raid.com/music/krizz_kaliko.html,benign

目标文件：
1. benign_urls.txt label为benign的url，每行一个
2. malware_urls.txt label为非benign的url，每行一个
"""

def process_csv_to_txt(input_csv, benign_output_txt, malware_output_txt):
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


if __name__ == '__main__':
    input_csv = 'kaggle_multi.csv'

    # 生成二分类数据集
    benign_output_txt = 'benign_urls.txt'
    malware_output_txt = 'malware_urls.txt'
    process_csv_to_txt(input_csv, benign_output_txt, malware_output_txt)
    print("完成生成")



