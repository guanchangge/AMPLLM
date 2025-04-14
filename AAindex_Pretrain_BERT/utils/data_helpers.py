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
from torch.utils.data.distributed import DistributedSampler
def get_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    """Read configuration information from the json configuration file"""
    dict = {}
    with open(json_file, 'r') as reader:
        text = reader.read()
    json_file = json.loads(text)
    for (key, value) in six.iteritems(json_file):
        dict[key] = value
    return dict

def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    Padding the elements in a List
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: Whether to put batch_size in the first dimension
        padding_value:
        max_len :
                When max_len = 50, it means padding the samples with a fixed length and truncating the excess；
                When max_len=None, it means padding the others with the length of the longest sample in the current batch；
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
    The purpose of this decorator is to cache the results of the data_process() method in the SQuAD dataset, so that it can be directly loaded next time it is used！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"Cache files {data_path} not exist, reprocess and cache！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"Cache file {data_path} exists, load cache file directly！")
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


class LoadMLMDataset:
    def __init__(self,
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=256,
                 pad_index=1,
                 is_sample_shuffle=True):
        self.tokenizer = tokenizer
        self.PAD_IDX = pad_index
        self.CLS_IDX = 0
        self.MASK_IDX = 32  # New MASK tag
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings

        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    def generate_batch(self, data_batch):
        batch_sentence, batch_label, regression_labels = [], [], []
        for (sen, label, regression_label) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(label)
            regression_labels.append(regression_label)
        batch_sentence = pad_sequence(batch_sentence, padding_value=self.PAD_IDX, max_len=self.max_sen_len)
        batch_label = pad_sequence(batch_label,max_len=self.max_sen_len)
        batch_label = batch_label.to(torch.long)
        regression_labels = torch.stack(regression_labels)
        #batch_label = torch.tensor(batch_label, dtype=torch.long)
        return  batch_sentence, batch_label,regression_labels

    @cache
    def data_process(self, filepath, postfix='cache'):
        raw_iter = open(filepath).readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            regression_labels = []
            for _ in range(1,554):
                regression_labels.append(float(line[_]))
            regression_labels = torch.tensor(regression_labels, dtype=torch.float32)
            s = line[0]
            tmp = [self.CLS_IDX] + self.tokenizer.encode(s)
            if len(tmp) > self.max_position_embeddings-1 :
                tmp = tmp[:self.max_position_embeddings-1 ]
            tmp += [2]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            #l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            masked_idx = torch.randperm(tensor_.size(0))[:int(0.15*tensor_.size(0))]
            masked_labels = torch.zeros(tensor_.size(0),dtype = torch.long)
            for idx in masked_idx:
                masked_labels[idx] = tensor_[idx].item()
            tensor_[masked_idx] = self.MASK_IDX
            data.append((tensor_,masked_labels,regression_labels))
        return data, max_len

    def load_data(self, train_file_path=None):
        postfix = str(self.max_sen_len)
        train_data, max_sen_len = self.data_process(filepath=train_file_path,postfix=postfix) 
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch,sampler=DistributedSampler(train_data))
        return train_iter

