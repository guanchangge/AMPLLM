import os
import torch
import pickle 
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import glob
import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef, auc, precision_recall_curve, roc_curve, confusion_matrix
import numpy as np
import random
# define the Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
# Define the Collate function to handlepadding for every batch 
def collate_fn(batch,max_seq_length=1028,cls=True): 
    
    # transform the features to tensor
    if cls:
        features = [item[2][1:len(item[1])+1].clone().detach() for item in batch]
    else:
        features = [item[2][0:len(item[1])].clone().detach() for item in batch]
    # obtain the labels
    labels = [int(item[0]) for item in batch]
    # Use the mask to indicate the non-padded part of the sequence
    masks = [torch.ones(len(seq), dtype=torch.bool) for seq in features]
    # truncate the sequence and mask (if exceed the max length)
    truncated_features = [seq[0:max_seq_length] if seq.size(0) > max_seq_length else seq for seq in features]
    truncated_masks = [seq[0:max_seq_length] if seq.size(0) > max_seq_length else seq for seq in masks]
    # padding the sequence and mask
    features_padded = pad_sequence(truncated_features, batch_first= True, padding_value= 0)
    masks_padded = pad_sequence(truncated_masks, batch_first= True, padding_value= False)
    # transform the labels to tensor
    labels_tensor = torch.tensor(labels,dtype=torch.long)
    return features_padded,labels_tensor,masks_padded
# preprocess data
def data_process(path,model_name,cls,max_seq_length=50):
    
    train_data = path + f'{model_name}_tr.pkl'
    test_data  = path + f'{model_name}_te.pkl'
    with open(train_data, 'rb') as f:
        train_data = pickle.load(f)
        train_dataset = CustomDataset(train_data)
    with open(test_data, 'rb') as f:
        test_data = pickle.load(f)
        test_dataset = CustomDataset(test_data)
    # create DataLoader and pass the custom collate function
    collate_fn_with_length = partial(collate_fn, max_seq_length = max_seq_length,cls=cls)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn_with_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn_with_length)
    feature_dim = train_data[0][2].size()[1]
    return train_loader, test_loader,feature_dim


class SequenceClassifier(nn.Module):
    def __init__(self, num_labels,hidden_size):
        super(SequenceClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.dense = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dense(self.dropout(hidden_states))
        return hidden_states
class CustomEncoderModel(nn.Module):
    def __init__(self,encoder_config):
        super(CustomEncoderModel, self).__init__()
        hidden_size = encoder_config.feature_dim
        num_labels = encoder_config.num_labels

        self.linear = torch.nn.Linear(hidden_size, 1)   # Input size: 200, Output size: 10
        self.softmax = torch.nn.Softmax(dim=1)
        self.classifier = SequenceClassifier(num_labels,hidden_size)
    def forward(self, features=None,mask=None):
        x = self.linear(features)
        x = self.softmax(x)
        output = features * x
        mask_expanded = mask.unsqueeze(-1).expand_as(features).float()
        masked_encoder_output = output * mask_expanded
        avg_pooled_output = masked_encoder_output.sum(dim=1)
        logits = self.classifier(avg_pooled_output)
        return logits

def evaluate_model(model,test_loader, test_features_gpu,test_labels_gpu,test_masks_gpu,device):
    model.eval()  
    true_labels = []
    predictions = []
    probabilities = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():  
        for batch_features, batch_labels, batch_mask in zip(test_features_gpu,test_labels_gpu,test_masks_gpu):
            batch_features, batch_labels, batch_mask = batch_features.to(device), batch_labels.to(device), batch_mask.to(device)
            outputs = model(features=batch_features, mask=batch_mask)
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item()
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()) 
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(batch_labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = accuracy_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    
    
    precision, recall, _ = precision_recall_curve(true_labels, probabilities)
    aupr = auc(recall, precision)
    
    
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc_score = auc(fpr, tpr)
    
    return {'SN': sn, 'SP': sp, 'ACC': acc, 'MCC': mcc, 'AUPR': aupr, 'AUC': auc_score,'Loss':total_loss/len(test_loader)}
def train_block(config,path,model_name,cls):
    record={}
    max_AUPR = 0
    train_loader, test_loader, feature_dim= data_process(path,model_name,cls,max_seq_length=config.max_seq_length)
    
    if feature_dim != config.feature_dim:
        config.feature_dim = feature_dim
    # Initialize your custom encoder model with the specified feature dimension and encoder configuration
    
    num_epochs = config.epochs
    train_features_gpu = []
    train_labels_gpu = []
    train_masks_gpu = []
    test_features_gpu = []
    test_labels_gpu = []
    test_masks_gpu = []
    for batch_features, batch_labels, batch_mask in test_loader:
        batch_features_gpu, batch_labels_gpu, batch_masks_gpu = batch_features.to(config.device), batch_labels.to(config.device), batch_mask.to(config.device)
        test_features_gpu.append(batch_features_gpu)
        test_labels_gpu.append(batch_labels_gpu)
        test_masks_gpu.append(batch_masks_gpu)
    for batch_features, batch_labels, batch_mask in train_loader:
        batch_features_gpu, batch_labels_gpu, batch_masks_gpu = batch_features.to(config.device), batch_labels.to(config.device), batch_mask.to(config.device)
        train_features_gpu.append(batch_features_gpu)
        train_labels_gpu.append(batch_labels_gpu)
        train_masks_gpu.append(batch_masks_gpu)
    
    
    custom_encoder_model = CustomEncoderModel(config)
    custom_encoder_model = custom_encoder_model.to(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(custom_encoder_model.parameters(), lr=config.lr)
    for epoch in range(num_epochs):
        record[f'epoch{epoch}']=[]
        true_labels = []
        predictions = []
        probabilities = []
        custom_encoder_model.train()
        total_loss = 0
        for batch_features, batch_labels, batch_mask in  zip(train_features_gpu, train_labels_gpu, train_masks_gpu):
            #batch_features, batch_labels, batch_mask = batch_features.to(config.device), batch_labels.to(config.device), batch_mask.to(config.device)
            # Forward pass
            optimizer.zero_grad()
            logits = custom_encoder_model(features=batch_features,mask=batch_mask)
            loss = loss_fn(logits, batch_labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()) # 使用Softmax获取概率分数
            _, predicted = torch.max(logits, 1)
            true_labels.extend(batch_labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
        # 计算指标
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        acc = accuracy_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)
    
        # 计算AUPR
        precision, recall, _ = precision_recall_curve(true_labels, probabilities)
        aupr = auc(recall, precision)
    
        # 计算AUC
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc_score = auc(fpr, tpr)
        metrics_train={'SN': sn, 'SP': sp, 'ACC': acc, 'MCC': mcc, 'AUPR': aupr, 'AUC': auc_score,'Loss':total_loss / len(train_loader)}
        record[f'epoch{epoch}'].append(metrics_train)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            metrics_value = evaluate_model(custom_encoder_model,test_loader,test_features_gpu,test_labels_gpu,test_masks_gpu,config.device)
            print(f"Accuracy on val {metrics_value['ACC']:.4f}")
            print(f"AUPR on val {metrics_value['AUPR']:.4f}")    
            print(f"Loss on val {metrics_value['Loss']:.4f}")
            if metrics_value['AUPR'] > max_AUPR:
                max_AUPR = metrics_value['AUPR']
                best_model = metrics_value
        record[f'epoch{epoch}'].append(metrics_value)
        
    return best_model,record

class ModelConfig:
    def __init__(self):
        self.num_labels = 2           # classification type
        self.max_seq_length = 100      # The max sequence length
        self.nhead= 8                # Number of attention heads
        self.dim_feedforward = 2048   # Dimension of the feedforward network model
        self.dropout=0.1              # Dropout rate
        self.activation= 'relu'       # Activation function
        self.feature_dim = 768        # Example dimension, corresponds to the output dimension of ESM1 model
        self.batch_size = 2048
        self.epochs = 5000            # Define the epochs
        self.model_val_per_epoch = 1  # Define the valuation step
        self.model_save_path = ''     # 
        self.lr = 2e-5                # define learning rate
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu') 

# set the random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if you use GPU, the CUDA random seed also need be set
config = ModelConfig()

model_list = ['esm2_t48_15B_UR50D','esm2_t36_3B_UR50D','esm2_t33_650M_UR50D','esm2_t30_150M_UR50D','esm2_t12_35M_UR50D','esm2_t6_8M_UR50D',
              'esm1v_t33_650M_UR90S_1','esm1v_t33_650M_UR90S_2','esm1v_t33_650M_UR90S_3','esm1v_t33_650M_UR90S_4','esm1v_t33_650M_UR90S_5',
              'esm1b_t33_650M_UR50S','esm1_t34_670M_UR50S','esm1_t34_670M_UR50D','esm1_t34_670M_UR100','esm1_t12_85M_UR50S','esm1_t6_43M_UR50S',
              'Tape_bert-base','ProtTrans_Prot_albert','ProtTrans_Prot_bert_bfd','ProtTrans_Prot_bert','ProtTrans_Prot_t5_xl_bfd','ProtTrans_Prot_t5_xl_uniref50',
              'ProtTrans_Prot_xlnet']
#model_list = ['AAindex_epoch25_full','AAindex_epoch50_partial','AAindex_epoch25_partial','AAindex_epoch50_full']
#path_way = '../Data/general_model/*'
#path_way = '/Data/data/*'
path_way = '/Data/AmPEP/*'
result_dict = {}
for database in glob.glob(path_way):
    if not os.path.isdir(database):
        continue
    database = database + '/'
    for model in model_list:
        print(model)
        if 'T5' in model or 'xlnet' in model:
            cls = False
        else:
            cls=True
        out_put = database.split('/')[-2] + '_' + model
        Best_value,record=train_block(config,database,model,cls)
        
        with open(f'{database}{model}_resul.txt','w') as w:
            for itter,value in record.items():
                text=itter + ','
                for item in value:
                    text += str(item['SN']) + ',' + str(item['SP']) + ',' + str(item['ACC']) + ',' + str(item['MCC']) + ',' + str(item['AUPR']) + ',' + str(item['AUC']) + ',' + str(item['Loss'])
                print(text)
                w.write(f'{text}\n')
        result_dict[out_put] = [tuple(list(Best_value.values()))]
        print(f'finised====={out_put}===={result_dict[out_put]}')
out_file = path_way.replace('*','') + 'statistic_results_weighted.csv'
#out_file = path_way.replace('*','') + 'AAindex_statistic_results_none_all_weighted.csv'
with open(out_file,'w') as w:
    for item in result_dict:
        # obtain model name 
        model_name = item
        print(model_name)
        # obtain the model result
        mean_result = result_dict[model_name][0]
        w.write(f"{model_name},Weight,")
        w.write(",".join(map(str, mean_result)))
        w.write("\n")
