import os


data_path = '/Work20/2020/wangshiquan/competiion/21bdci/data/train.txt'
data_enhance_path = '/Work20/2020/wangshiquan/competiion/21bdci/data/train_enhance.txt'

def data_deal(data_path, data_enhance_path):
    with open(data_path, 'r', encoding='utf-8') as f1:
        with open(data_enhance_path, 'w', encoding='utf-8') as f2:
            for line in f1.readlines():
                line = line.strip()
                if line:
                    example = line.split('\t')
                    temp = example[0]
                    example[0] = example[1]
                    example[1] = temp
                    f2.writelines(example[0] + '\t' + example[1] + '\t' +example[2] + '\n')

# if not os.path.exists(data_enhance_path):
#     os.mkdir(data_enhance_path)
def data_deal2(data_path, data_enhance_path):
    with open(data_path, 'r', encoding='utf-8') as f1:
        with open(data_enhance_path, 'a', encoding='utf-8') as f2:
            for line in f1.readlines():
                line = line.strip()
                if line:
                    example = line.split('\t')
                    f2.writelines(example[0] + '\t' + example[1] + '\t' +example[2] + '\n')

data_deal(data_path, data_enhance_path)
data_deal2(data_path, data_enhance_path)