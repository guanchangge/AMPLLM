import sys
sys.path.append('../')
from utils.data_helpers import LoadMLMDataset, get_json_file
from torch.utils.tensorboard import SummaryWriter
from utils import logger_init
from utils.set_seed import set_seed
from transformers import BertTokenizer
from model import BertForSequenceRepresentation
from transformers import AdamW
import logging
import torch
import os
import time
import esm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch.distributed as dist
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import LambdaLR
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
class ModelConfig:
    """
    Model configure file
    """
    def __init__(self):
        # file path
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.train_file_path = os.path.join(self.dataset_dir, 'train_all_set.csv')
        self.test_file_path = os.path.join(self.dataset_dir, 'test_set.csv')
        self.data_name = 'AAindex_finetune'
        # Cache/log path
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        # Hyperparameters (often modified)
        self.split_sep = ','   # Separator, used for data processing (generally not used)
        self.is_sample_shuffle = False  # Shuffle the training set
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pad_token_id = 1  #!!!!
        self.batch_size = 32
        self.max_sen_len = None
        self.num_labels = 33   # Classification category (easy to ignore)
        self.learning_rate = 1e-4
        self.epochs = 100
        self.model_val_per_epoch = 1  # The model is evaluated every few rounds
        self.random_seed = 42
        self.max_position_embeddings = 128
        
        # Model save address/tensorbroad log address/logging log address
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}_seed{self.random_seed}.bin')
        self.model_save_R_Square_path = os.path.join(self.model_save_dir, f'model_{self.data_name}_seed{self.random_seed}_R_Square.bin')
        self.writer = SummaryWriter(f"runs/{self.data_name}" + '_seed' + str(self.random_seed))  
        logger_init(log_file_name=self.data_name + '_seed' + str(self.random_seed), log_level=logging.INFO, log_dir=self.logs_save_dir)
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)
total_steps=200
def lr_lambda(epoch):
    warmup_steps = 5
    if epoch < warmup_steps:
        return float(epoch) / float(max(1, warmup_steps))
    else:
        return max(0.1, 1.0 - float(epoch - warmup_steps) / float(max(1, 0.9 * total_steps - warmup_steps)))
def calculate_r_squared(y_true, y_pred):
    res = torch.sum((y_true - y_pred)**2)
    tot = torch.sum((y_true - torch.mean(y_true))**2)
    R_square = 1 - (res / (tot+1e-8))
    return R_square
def ddp_setup(rank, world_size):
    """
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "14233"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
#def train(config):
def train(rank: int, world_size: int):
    config = ModelConfig()
    ddp_setup(rank,world_size)
    model = BertForSequenceRepresentation(config).to(rank)
    bert_tokenize = model.tokenize
    '''
    for param in model.bert.parameters():
        param.requires_grad = False

    for param in model.bert.layers[-15:].parameters():
        param.requires_grad = True
    '''
    logging.info(f"{model}")
    #model = model.to(config.device)
    model = DDP(model, device_ids=[rank])
    '''
    if os.path.exists(config.model_save_path):
        loaded_paras = torch.load(config.model_save_path,map_location=config.device)
        print(loaded_paras)
        model.load_state_dict(loaded_paras)
        logging.info("## Successfully loaded the existing model for additional training......")
    print('========================')
    '''
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=2)
    #scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1,20,40,80], gamma=0.1)
    data_loader = LoadMLMDataset(tokenizer=bert_tokenize,
                                            batch_size=config.batch_size,
                                            max_sen_len=config.max_sen_len,
                                            split_sep=config.split_sep,
                                            max_position_embeddings=config.max_position_embeddings,
                                            pad_index=config.pad_token_id,
                                            is_sample_shuffle=config.is_sample_shuffle)
    train_iter = data_loader.load_data(config.train_file_path)
    test_iter = data_loader.load_data(config.test_file_path)

    best_loss =100000000
    max_R_Square =-10000000000000
    save_steps = 10000000000  # Set the step size for saving the model
    for epoch in range(config.epochs):
        model.train()
        losses = 0
        res_losses = 0
        start_time = time.time()
        for idx, (sample,masked_labels,regression_label) in enumerate(train_iter):
            global_iter_num = epoch * len(train_iter) + idx + 1  # Calculate the current step from the beginning of training (global iteration number)
            #sample = sample.transpose(0,1).to(config.device)  # [src_len, batch_size]
            sample = sample.transpose(0,1).to(rank)
            #label = label.to(config.device)
            #masked_labels = masked_labels.transpose(0,1).to(config.device)
            #regression_label = regression_label.to(config.device)
            masked_labels = masked_labels.transpose(0,1).to(rank)
            regression_label = regression_label.to(rank)
            padding_mask = (sample != data_loader.PAD_IDX) 
            loss, logits,res_loss = model(input_ids=sample,attention_mask=padding_mask,token_type_ids=None,position_ids=None,labels=None,masked_labels=masked_labels,regression_labels=regression_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            res_losses += res_loss.item()
            
            if idx % 1000 == 0:
                logging.info(f"Epoch: [{epoch}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f} {res_loss.item():.3f}")
                config.writer.add_scalar('Training/Loss', loss.item(), global_iter_num)
            if idx % save_steps == 0 and rank == 0:
            # Save the model with a certain training step length
                torch.save(model.state_dict(), f'checkpoint_{epoch}_{idx}.bin')
        end_time = time.time()
        train_loss = losses / len(train_iter)
        res_losses = res_losses/ len(train_iter)
        scheduler.step()
        if train_loss < best_loss and rank==0:
            best_loss = train_loss
            # save model
            torch.save(model.state_dict(), config.model_save_path)
            logging.info(f"Saved model with MLM Loss: {train_loss:.3f}")
            logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f} {res_losses:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for x, _, y in test_iter:
                    x, y= x.transpose(0,1).to(rank), y.transpose(0,1).to(rank)
                    padding_mask = (x != data_loader.PAD_IDX)
                    regression_outputs = model(input_ids=x,attention_mask=padding_mask)
                    for i in range(553):
                        predicted = regression_outputs[i].cpu().numpy()
                        true_value = y[i].cpu().numpy()
                        R = r2_score(true_value,predicted.reshape(-1))
                        total_loss += R
                    
                ave_total_loss = total_loss/553/len(test_iter)

                    
            logging.info(f"Average R Square report {ave_total_loss}")
            config.writer.add_scalar('valid/R Square', ave_total_loss, epoch)
            if ave_total_loss > max_R_Square and rank==0:
                max_R_Square = ave_total_loss
                torch.save(model.state_dict(), config.model_save_R_Square_path)


if __name__ == '__main__':
    model_config = ModelConfig()
    set_seed(model_config.random_seed)
    world_size = 6
    mp.spawn(train, args=(world_size,),nprocs=world_size)
    #train(model_config)
