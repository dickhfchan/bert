'''从pip下载bert-serving-server模型
下载语句:pip install bert-serving-server
pip install bert-serving-client
然后将下载的中文编码chinese_L-12_H-768_A-12放到bert-serving模型中启动
在cmd中输入 : bert-serving-start -model_dir (放中文编码的路径) -num_worker=1
启动完成就可以进行调用,将文字转换成词向量'''

from bert_serving.client import BertClient
import numpy as np
import pandas as pd
import time
import tensorflow as tf


bc=BertClient(port=5555,port_out=5556)
def ner_test():
    with BertClient(show_server_config=False, check_version=False, check_length=False) as bc:
        start_t = time.perf_counter()
        str1 = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
        str1 = list(str1)
        rst = bc.encode([str1], is_tokenized=True)
        print('rst:', rst)
        print(len(rst[0]))
        print(time.perf_counter() - start_t)
# file=pd.read_csv("dd.txt")
# data=file.to_string()
# for line in data:
#     result.append(line.strip("\n"))
# class_test()
ner_test()
