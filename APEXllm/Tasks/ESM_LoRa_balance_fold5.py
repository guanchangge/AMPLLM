import sys

sys.path.append('../')

from utils.data_helpers import LoadlassificationDataset, get_json_file
from torch.utils.tensorboard import SummaryWriter
from utils import logger_init
from utils.set_seed import set_seed
from transformers import BertTokenizer
from model import BertForSequenceClassification
from transformers import AdamW
import logging
import torch
import os
import time
import esm
from peft import LoraConfig, get_peft_model,PeftModel
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
class ModelConfig:
    """
    Model configure file
    """
    def __init__(self):
        # file path
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data/ESM_LoRa/5fold/')
        self.train_file_path = os.path.join(self.dataset_dir, 'balance_train_fold_5.csv')
        self.val_file_path = os.path.join(self.dataset_dir, 'balance_val_fold_5.csv')   
        self.test_file_path = os.path.join(self.dataset_dir, 'balance_val_fold_5.csv')
        self.data_name = 'ESM2_LoRa_balance_fold5'
        # 缓存/日志路径
        self.model_save_dir = os.path.join(self.project_dir, 'final_model')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        # 超参数(常修改)
        self.split_sep = ','   # 分割符,处理数据使用(一般不用)
        self.is_sample_shuffle = True  # 训练集打乱
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.pad_token_id = 1  #!!!!
        self.batch_size = 4
        self.max_sen_len = None
        self.num_labels = 2   # 分类类别(易忽略)
        self.learning_rate = 1e-5
        self.epochs = 10
        self.model_val_per_epoch = 1  # 模型每个几轮进行评测
        self.random_seed = 42
        self.max_position_embeddings = 100 
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}_seed{self.random_seed}.bin')
        self.writer = SummaryWriter(f"runs/{self.data_name}" + '_seed' + str(self.random_seed))  
        logger_init(log_file_name=self.data_name + '_seed' + str(self.random_seed), log_level=logging.INFO, log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)

def initialize_peft(
    model,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.3,):
    
    lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model
def train(config):
    model = BertForSequenceClassification(config)
    model = initialize_peft(model)
    logging.info(f"{model}")
    model = model.to(config.device)
    bert_tokenize = model.tokenize
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9,0.98),eps=1e-08, lr=config.learning_rate,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001,min_lr=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    data_loader = LoadlassificationDataset(tokenizer=bert_tokenize,
                                            batch_size=config.batch_size,
                                            max_sen_len=config.max_sen_len,
                                            split_sep=config.split_sep,
                                            max_position_embeddings=config.max_position_embeddings,
                                            pad_index=config.pad_token_id,
                                            is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        model.train()
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            global_iter_num = epoch * len(train_iter) + idx + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
            sample = sample.transpose(0,1).to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample != data_loader.PAD_IDX)
            loss, logits = model(input_ids=sample,attention_mask=padding_mask,token_type_ids=None,position_ids=None,labels=label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
            batch_acc = (logits.argmax(1) == label).float().mean()
            if idx % 1000 == 0:
                logging.info(f"Epoch: [{epoch}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {batch_acc:.3f}")
                config.writer.add_scalar('Training/Loss', loss.item(), global_iter_num)
                config.writer.add_scalar('Training/Accuracy', batch_acc, global_iter_num)
        # 每隔model_ver_per_step评估模型在验证集上效果
        if (epoch+1) % config.model_val_per_epoch == 0:
            val_acc,val_mcc,val_SP,val_SN, cls_report,auc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
            _,_,_,_,_,_ = evaluate(test_iter, model, config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {val_acc:.4f}")
            logging.info(f"MCC on val {val_mcc:.4f}")
            logging.info(f"classification report {cls_report}")
            config.writer.add_scalar('valid/Accuracy', val_acc, epoch)
            if  auc> max_acc:
                max_acc = auc
                torch.save(model.state_dict(), config.model_save_path)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        scheduler.step(train_loss)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.6f}, Epoch time = {(end_time - start_time):.3f}s")
        

def inference(config):
    model = BertForSequenceClassification(config)
    model = initialize_peft(model)
    model_weights = torch.load(config.model_save_path,map_location=config.device)
    model.load_state_dict(model_weights)
    logging.info(f"## 成功载入{config.model_save_path}已有模型，进行预测......")
    model = model.merge_and_unload()
    del model_weights
    torch.cuda.empty_cache()
    bert_tokenize = model.tokenize
    data_loader = LoadlassificationDataset(tokenizer=bert_tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    acc,Mcc,SP,SN,report,auc = evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    logging.info(f"Acc on test:{acc:.4f}")
    logging.info(f"MCC,SP,SN on test:{Mcc:.4f} {SP:.4f} {SN:.4f}")
    logging.info(f"classification report:{report}")
    logging.info(f"auc:{auc}")


def evaluate(data_iter, model, device, PAD_IDX):
    from sklearn.metrics import f1_score, accuracy_score, classification_report,matthews_corrcoef,confusion_matrix,roc_auc_score,precision_recall_curve,auc
    import numpy as np
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        real_res = []
        pred_res = []
        prob = []
        for x, y in data_iter:
            x, y = x.transpose(0,1).to(device), y
            padding_mask = (x != PAD_IDX)
            logits = model(
                input_ids=x,
                attention_mask=padding_mask)
            probabilities = torch.sigmoid(logits)
            logits = logits.detach().cpu().numpy()
            pred_res_ = np.argmax(logits, axis=1).flatten()
            pred_res = np.concatenate((pred_res, pred_res_))

            label_ids = y.to('cpu').numpy()
            real_res = np.concatenate((real_res, label_ids))
            probabilities = probabilities.to('cpu').numpy()
            prob.append(probabilities)
        prob = np.concatenate(prob)[:, 1]
        auc_score = roc_auc_score(real_res, prob)
        accuracy = accuracy_score(real_res, pred_res)  
        Mcc = matthews_corrcoef(real_res, pred_res)
        classification_report = classification_report(real_res, pred_res, digits=4)  
        TN, FP, FN, TP = confusion_matrix(real_res, pred_res).ravel()
        SN = TP/(TP+FN)
        SP = TN/(TN+FP)
        print(auc_score)
        print(accuracy)
        precision, recall, _ = precision_recall_curve(real_res, prob)
        aupr = auc(recall, precision)
        print(aupr)
        model.train()
        return accuracy,Mcc,SP,SN, classification_report,auc_score


if __name__ == '__main__':
    model_config = ModelConfig()
    set_seed(model_config.random_seed)
    #train(model_config)
    inference(model_config)
