import pickle
import glob
import torch
import numpy as np
from tape import ProteinBertModel, TAPETokenizer
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


def Tape_extract_representation(model_name,seq_file,device,tokenize):
    bert = model_name
    batch_converter = tokenize
    # Select GPU
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    bert = bert.to(device)
    bert.eval()
    
    seq_lst,label_lst = extract_seq(seq_file)
    # Extract representation of sequences and save them.
    feature_list = []
    with torch.no_grad():
        for seq,label in zip(seq_lst,label_lst):
            batch = seq
            batch_tokens = torch.tensor(np.array([batch_converter.encode(batch)]))
            batch_labels = label
            batch_tokens = batch_tokens.to(device, dtype=torch.long)
            results = bert(batch_tokens)
            token_representations = results[0]
            feature_list.append((batch_labels,batch,token_representations[0].cpu()))
    return feature_list

model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
device = 4
path_way_list = ['../Data/data/*','../Data/AmPEP/*','../Data/general_model/*']

for path_way in path_way_list:
    databases = glob.glob(path_way)
    for database in databases:
        database = database + '/*'
        for data_set in glob.glob(database):
            if data_set.split('/')[-1].split('.')[-1] == 'csv':
                print(data_set)
                feature_list = Tape_extract_representation(model,data_set,device,tokenizer)
                if 'tr' in  data_set.split('/')[-1]:
                    feature_file = database.replace('*','')  + f'Tape_bert-base_tr.pkl'
                elif 'ex' in data_set.split('/')[-1]:
                    feature_file = database.replace('*','')  + f'Tape_bert-base_ex.pkl'
                else:
                    feature_file = database.replace('*','')  + f'Tape_bert-base_te.pkl'
                with open(feature_file, 'wb') as f:
                    pickle.dump(feature_list, f)
                print(f'Finishing ====={feature_file}')
