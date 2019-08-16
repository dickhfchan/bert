#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
from bert import modeling
from bert import tokenization
import pandas as pd
import os

# 一、加载BERT模型
# 这里是下载下来的bert配置文件
data_root = 'wwm_uncased_L-24_H-1024_A-16/'
bert_config_file = data_root + 'bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
#  创建bert的输入
input_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="segment_ids")

# 创建bert模型
model = modeling.BertModel(
    config=bert_config,
    is_training=True,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
)

#bert模型参数初始化的地方
init_checkpoint = "wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt"
use_tpu = False
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型

(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                       init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# 二、获取BERT模型的输出
output_layer = model.get_sequence_output()# 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个

# output_layer = model.get_pooled_output() # 这个获取句子的output

# 三、获取BERT模型的输入
def convert_single_example( max_seq_length,
                           tokenizer,text_a,text_b=None):
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_a)# 这里主要是将中文分字
  if tokens_b:
    # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
    # 因为要为句子补上[CLS], [SEP], [SEP]
    tokens_a+tokens_b<=max_seq_length-3
    # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
  # (a) 两个句子:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) 单个句子:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 这里 "type_ids" 主要用于区分第一个第二个句子。
  # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
  # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
  # 创建mask
  input_mask = [1] * len(input_ids)
  # 对于输入进行补0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids ,tokens# 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数


# 下面开始调用分词并且生成输入数据
# file=pd.read_csv("data1.txt")
# data=file.to_string()
data='{"0":{"year":"2019","date":"2019-05-08","url":"https:\/\/www.bloomberg.com\/news\/articles\/2019-05-07\/u-s-cotton-production-expected-to-reach-highest-in-14-years?srnd=constap","body":"U.S. cotton output may swing back to the highest since 2005.The 2019-2020 U.S. crop will expand to 21.8 million bales from 18.4 million the previous season, according to the average estimate among analysts surveyed by Bloomberg ahead of the U.S. World Agricultural Supply and Demand Estimates report. The USDA is scheduled to release the report Friday at noon in Washington. A bale weighs 480 pounds (218 kilograms).U.S. & World Cotton Production, Inventory Survey Before WASDEIn March, the USDA forecast that cotton acres would dip to 13.8 million for the 2019-2020 season from 14.1 million a year earlier. Still, anecdotal evidence in states including Texas and Oklahoma suggests the fiber will be sown on 14 million acres nationally, according to John Robinson, a professor of agricultural economics at Texas A&M University.Relatively low prices for competing crops and rainy weather causing planting delays for corn may make the fiber an attractive alternative, said Robinson, who estimates U.S. cotton production will reach 23 million bales.","link_name":"Generic 1st "CT" Future","tags":"Generic 1st "CT" Future,Washington,Texas,Oklahoma,Economics,Markets,Futures Markets,Commodity Futures,Secondary Brand: markets,Secondary Brand: business,Franchise: economics,Region: Global,Page: article","title":"U.S. Cotton Production Expected to Reach Highest in 14 Years"},"1":'
# data2='how are you?I am fine. ajfklsjdflkjajwo'
print(len(data))
vocab_file="wwm_uncased_L-24_H-1024_A-16/vocab.txt"
token = tokenization.FullTokenizer(vocab_file=vocab_file)
input_ids,input_mask,segment_ids,tokens = convert_single_example(len(data),token,data)

print('tokens:',tokens)
print('input_ids:',input_ids)
print('input_mask:',input_mask)
print('segment_ids:',segment_ids)
