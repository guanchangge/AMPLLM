from asyncore import file_dispatcher
from tkinter.tix import Tree
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import numpy as np
import collections
import six
import copy
import random

def get_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    """从json配置文件读取配置信息"""
    dict = {}
    with open(json_file, 'r') as reader:
        text = reader.read()
    json_file = json.loads(text)
    for (key, value) in six.iteritems(json_file):
        dict[key] = value
    return dict

def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def cache(func):
    """
    本修饰器的作用是将SQuAD数据集中data_process()方法处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


def process_input(seq):
    """
    Separate the input sequence with spaces   (eg. "ABCD"--->"A B C D")
    param: seq: input sequence
    return: Sequence separated by spaces
    """
    pro_seq = ''
    for i in range(len(seq)):
        if i == 0:
            pro_seq += seq[i]
        else:
            pro_seq += " " + seq[i]
    return pro_seq




class LoadlassificationDataset:
    def __init__(self,
                tokenizer=None,
                batch_size=32,
                max_sen_len=None,
                split_sep='\n',
                max_position_embeddings=256,
                pad_index=1,
                is_sample_shuffle=True):
        """
        :param vocab_path: 本地词表vocab.txt的路径
        :param tokenizer:
        :param batcg_size:
        :param max_sen_len: 在对每个batch进行处理时的配置
                            当max_sen_len = None时，即每个batch中最长样本长度为标准，对其进行padding
                            当max_sen_len = 'same'时，以整个数据集中最长样本为标准，对其他进行padding
                            当max_sen_len = 50， 表示以某个固定长度符样本进行padding，多余的截断；
        :param split_sep: 文本和标签之前的分隔符，默认为'\t'
        :param max_position_embeddings: 指定最大样本长度，超过长度的部分将截取掉
        :param is_sample_shuffle: 是否打乱训练集样本（只针对训练集）
                    在后续构造DataLoader时，验证集和测试集均指定为了固定顺序（即不进行打乱），修改程序时请勿进行打乱
                    因为当shuffle为True时，每次通过for循环遍历data_iter时样本的顺序都不一样，这回导致在模型预测时
                    返回的标签顺序和原始的顺序不一样，不方便处理。
        """
        self.tokenizer = tokenizer
        self.PAD_IDX = pad_index
        self.SEP_IDX = 2
        self.CLS_IDX = 0
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings

        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    @cache
    def data_process(self, filepath, postfix='cache'):
        """
        将每一句话的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param filepath: 数据集路径
        :return:
        """
        raw_iter = open(filepath).readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            s, l = line[0], line[1]
            #if s[0]=='M':
                #s=s[1:]
            tmp = [self.CLS_IDX] + self.tokenizer.encode(s)
            if len(tmp) > self.max_position_embeddings-1 :  # 截断
                tmp = tmp[:self.max_position_embeddings-1 ]  # BERT预训练模型只取前512字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))  # 保存最长序列长度
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process(filepath=test_file_path, postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(filepath=train_file_path,
                                                    postfix=postfix)  # 得到处理好的所有样本
        
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(filepath=val_file_path,
                                        postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter
        

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 对一个batch中的每个样本进行处理
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size, max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len) 
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return  batch_sentence, batch_label

