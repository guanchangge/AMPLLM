import torch
from transformers import AlbertModel, AlbertTokenizer, BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import pickle
import glob
import numpy as np
from tape import ProteinBertModel, TAPETokenizer
from transformers import T5EncoderModel, T5Tokenizer,XLNetModel, XLNetTokenizer
import gc
#read sequence data
def extract_seq(file_seq):
    seq_lst = []
    label_lst = []
    with open (file_seq,'r') as f:
        for line in f:
            seq = line.split(',')[0]
            label = line.split(',')[1]
            seq_lst.append(seq)
            label_lst.append(label)
    return seq_lst,label_lst



def Prot_albert(data_set,device,database):
    device = device
    tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
    model = AlbertModel.from_pretrained("Rostlab/prot_albert")
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)    
    model = model.eval()
    seq_lst,label_lst = extract_seq(data_set)
    
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            input_seq = [" ".join(seq)]
            input_seq = [re.sub(r"[UZOB]", "X", item) for item in input_seq]
            ids = tokenizer.batch_encode_plus(input_seq, add_special_tokens=True, pad_to_max_length=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
            embedding = embedding.cpu() 
            feature_list.append((label,seq,embedding[0]))
    if 'tr' in  data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_albert_tr.pkl'
    elif 'ex' in data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_albert_ex.pkl'
    else:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_albert_te.pkl'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_list, f)              
    print(f'Finishing ====={feature_file}')

def Prot_bert_bfd(data_set,device,database):
    device = device
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_lst,label_lst = extract_seq(data_set)
   
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            input_seq = [" ".join(seq)]
            input_seq = [re.sub(r"[UZOB]", "X", item) for item in input_seq]
            ids = tokenizer.batch_encode_plus(input_seq, add_special_tokens=True, pad_to_max_length=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
            embedding = embedding.cpu()
            feature_list.append((label,seq,embedding[0]))
    if 'tr' in  data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_bert_bfd_tr.pkl'
    elif 'ex' in data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_bert_bfd_ex.pkl'
    else:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_bert_bfd_te.pkl'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_list, f)         
    print(f'Finishing ====={feature_file}')

def Prot_bert(data_set,device,database):
    device = device
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_lst,label_lst = extract_seq(data_set)
   
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            input_seq = [" ".join(seq)]
            input_seq = [re.sub(r"[UZOB]", "X", item) for item in input_seq]
            ids = tokenizer.batch_encode_plus(input_seq, add_special_tokens=True, pad_to_max_length=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
            embedding = embedding.cpu()
            feature_list.append((label,seq,embedding[0]))
    if 'tr' in  data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_bert_tr.pkl'
    elif 'ex' in data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_bert_ex.pkl'
    else:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_bert_te.pkl'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_list, f)    
    print(f'Finishing ====={feature_file}')


def Prot_t5_xl_bfd(data_set,device,database):
    device = device
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")
    gc.collect()
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_lst,label_lst = extract_seq(data_set)
    
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            input_seq = [" ".join(seq)]
            input_seq = [re.sub(r"[UZOB]", "X", item) for item in input_seq]
            ids = tokenizer.batch_encode_plus(input_seq, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu()
            feature_list.append((label,seq,embedding[0]))
    if 'tr' in  data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_t5_xl_bfd_tr.pkl'
    elif 'ex' in data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_t5_xl_bfd_ex.pkl'
    else:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_t5_xl_bfd_te.pkl'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_list, f) 
    print(f'Finishing ====={feature_file}')

 
def Prot_t5_xl_uniref50(data_set,device,database):
    device = device
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_lst,label_lst = extract_seq(data_set)
   
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            input_seq = [" ".join(seq)]
            input_seq = [re.sub(r"[UZOB]", "X", item) for item in input_seq]
            ids = tokenizer.batch_encode_plus(input_seq, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu()
            feature_list.append((label,seq,embedding[0]))
    if 'tr' in  data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_t5_xl_uniref50_tr.pkl'
    elif 'ex' in data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_t5_xl_uniref50_ex.pkl'
    else:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_t5_xl_uniref50_te.pkl'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_list, f) 
    print(f'Finishing ====={feature_file}')

def Prot_xlnet(data_set,device,database):
    device = device
    tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
    xlnet_men_len = 512
    model = XLNetModel.from_pretrained("Rostlab/prot_xlnet",mem_len=xlnet_men_len)
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_lst,label_lst = extract_seq(data_set)
    
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            input_seq = [" ".join(seq)]
            input_seq = [re.sub(r"[UZOBX]", "<unk>", item) for item in input_seq]
            ids = tokenizer.batch_encode_plus(input_seq, add_special_tokens=True, pad_to_max_length=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            with torch.no_grad():
                output = model(input_ids=input_ids,attention_mask=attention_mask,mems=None)
                embedding = output.last_hidden_state
            embedding = embedding.cpu()
            feature_list.append((label,seq,embedding[0]))
    if 'tr' in  data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_xlnet_tr.pkl'
    elif 'ex' in data_set.split('/')[-1]:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_xlnet_ex.pkl'
    else:
        feature_file = database.replace('*','')  + f'ProtTrans_Prot_xlnet_te.pkl'
    with open(feature_file, 'wb') as f:
        pickle.dump(feature_list, f)

    print(f'Finishing ====={feature_file}')


path_way_list = ['../Data/data/*','../Data/AmPEP/*','../Data/general_model/*']
device = 5
for path_way in path_way_list:
    databases = glob.glob(path_way)
    for database in databases:
        database = database + '/*'
        for data_set in glob.glob(database):
            if data_set.split('/')[-1].split('.')[-1] == 'csv':
                print(data_set)
                Prot_albert(data_set,device,database)
                Prot_bert_bfd(data_set,device,database)
                Prot_bert(data_set,device,database)
                Prot_t5_xl_bfd(data_set,device,database)
                Prot_t5_xl_uniref50(data_set,device,database) 
                Prot_xlnet(data_set,device,database)
        
