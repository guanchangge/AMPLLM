# Import necessary libraries
import os
import logging
import pickle
import numpy as np
import torch
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, accuracy_score, matthews_corrcoef
model_list = ['esm2_t48_15B_UR50D','esm2_t36_3B_UR50D','esm2_t33_650M_UR50D','esm2_t30_150M_UR50D','esm2_t12_35M_UR50D','esm2_t6_8M_UR50D',
              'esm1v_t33_650M_UR90S_1','esm1v_t33_650M_UR90S_2','esm1v_t33_650M_UR90S_3','esm1v_t33_650M_UR90S_4','esm1v_t33_650M_UR90S_5',
              'esm1b_t33_650M_UR50S','esm1_t34_670M_UR50S','esm1_t34_670M_UR50D','esm1_t34_670M_UR100','esm1_t12_85M_UR50S','esm1_t6_43M_UR50S',
              'Tape_bert-base','ProtTrans_Prot_albert','ProtTrans_Prot_bert_bfd','ProtTrans_Prot_bert','ProtTrans_Prot_t5_xl_bfd',
              'ProtTrans_Prot_t5_xl_uniref50','ProtTrans_Prot_xlnet']
#model_list = ['AAindex_epoch25_partial','AAindex_epoch50_partial','AAindex_epoch25_full','AAindex_epoch50_full']
def data_load(data_file):
    with open(data_file,'rb') as f:
        data = pickle.load(f)
    return data
def data_extraction_bert(data):
    label_lst = []
    #seq_lst = []
    cls_feature_lst =[]
    mean_feature_lst = []
    for label,seq,feature in data:
        label_lst.append(int(label.strip()))
        #seq_lst.append(seq)
        cls_feature_lst.append(feature[0].numpy())
        mean_feature_lst.append((torch.mean(feature[1:len(seq)+1],axis=0)).numpy())
    return np.array(cls_feature_lst),np.array(mean_feature_lst),np.array(label_lst)
def data_extraction_t5(data):
    label_lst = []
    #seq_lst = []
    mean_feature_lst = []
    for label,seq,feature in data:
        label_lst.append(int(label.strip()))
        #seq_lst.append(seq)
        mean_feature_lst.append((torch.mean(feature[:len(seq)],axis=0)).numpy())
    return np.array(mean_feature_lst),np.array(label_lst)

def Calculate_model(model,X_train, y_train,X_test,y_test): 
    # Train the model
    model = model
    model.fit(X_train, y_train)
    # Predict probabilities and classes on the testing set
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    aupr = auc(recall, precision)
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_test, y_pred)
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    return sensitivity,specificity,accuracy,mcc,aupr,auc_score
result_dict = {}
# Initialize logistic regression model
#path_way = '../Data/general_model/*'
#path_way = '/Data/data/*'
path_way = '/Data/AmPEP/*'
for database in glob.glob(path_way):
    if os.path.isdir(database):
        database = database + '/'
        for model_name in model_list:
            model = LogisticRegression(max_iter=5000,penalty=None,random_state=42)
            out_put = database.split('/')[-2] + '_' + model_name
            train_data = database + model_name + '_tr.pkl'
            train_data = data_load(train_data)
            test_data = database + model_name + '_te.pkl'
            test_data = data_load(test_data)
            print(model_name)
            if 't5'  not in model_name and 'xlnet' not in model_name:
                print(1)
                X_train_cls,X_train_mean,y_train = data_extraction_bert(train_data)
                X_test_cls, X_test_mean,y_test = data_extraction_bert(test_data)
                sensitivity_cls,specificity_cls,accuracy_cls,mcc_cls,aupr_cls,auc_score_cls = Calculate_model(model,X_train_cls,y_train,X_test_cls, y_test)
                sensitivity_mean,specificity_mean,accuracy_mean,mcc_mean,aupr_mean,auc_score_mean = Calculate_model(model,X_train_mean,y_train,X_test_mean, y_test)
                result_dict[out_put] =[(sensitivity_cls,specificity_cls,accuracy_cls,mcc_cls,aupr_cls,auc_score_cls),(sensitivity_mean,specificity_mean,accuracy_mean,mcc_mean,aupr_mean,auc_score_mean)]

            else:
                print(0)
                X_train_mean,y_train = data_extraction_t5(train_data)
                X_test_mean, y_test = data_extraction_t5(test_data)
                sensitivity_mean,specificity_mean,accuracy_mean,mcc_mean,aupr_mean,auc_score_mean = Calculate_model(model,X_train_mean,y_train,X_test_mean, y_test)
                result_dict[out_put] =[(sensitivity_mean,specificity_mean,accuracy_mean,mcc_mean,aupr_mean,auc_score_mean)]
            print(f'finised====={out_put}===={result_dict[out_put]}')
out_file = path_way.replace('*','') + 'statistic_results_none.csv'
#out_file = path_way.replace('*','') + 'AAindex_statistic_results_none_full_50.csv'
with open(out_file,'w') as w:
    for item in result_dict:
        # obtain the model name
        model_name = item
        print(model_name)
        
        # obtain the result of the model
        if 't5' not in model_name and 'xlnet' not in model_name:
            print(1)
            cls_result = result_dict[model_name][0]
            mean_result = result_dict[model_name][1]
            # write the model name and results
            w.write(f"{model_name},Classification,")
            w.write(",".join(map(str, cls_result)))
            w.write("\n")
            w.write(f"{model_name},Mean,")
            w.write(",".join(map(str, mean_result)))
            w.write("\n")
        else:
            print(1)
            mean_result = result_dict[model_name][0]
            w.write(f"{model_name},Mean,")
            w.write(",".join(map(str, mean_result)))
            w.write("\n")



