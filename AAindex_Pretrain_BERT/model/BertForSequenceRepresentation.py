#Model definition
import torch.nn as nn
import torch
import esm

class BertForSequenceRepresentation(nn.Module):
    def __init__(self, config):
        super(BertForSequenceRepresentation, self).__init__()
        self.num_labels = config.num_labels
        self.bert,self.tokenize = esm.pretrained.esm2_t36_3B_UR50D()
        hidden_size = 2560
        del self.bert.contact_head
        #self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.regression_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(553)])
    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                masked_labels=None,
                regression_labels=None):
        """
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        output = self.bert(input_ids,repr_layers=[36])  # [batch_size,hidden_size]
        logits = output ['logits']  # [batch_size, num_label]
        regression_outputs = []
        for regression_layer in self.regression_layers:
            regression_output = regression_layer(output['representations'][36][:, 0, :])
            regression_outputs.append(regression_output) 
        if masked_labels is not None :
            # Convert logits from a 3D tensor to a 2D tensor
            logits = logits.view(-1, logits.shape[-1])
            # Convert masked_labels to a 1D tensor
            #masked_labels = masked_labels.transpose(0, 1)
            masked_labels = masked_labels.reshape(-1).to(torch.long)
            # Find the index of the mask position
            non_masked_indices = (masked_labels != 0).nonzero().squeeze()
            # Keep only the logits and labels of the masked positions
            masked_logits = logits[non_masked_indices]
            masked_labels = masked_labels[non_masked_indices]
            loss_fct = nn.CrossEntropyLoss()
            masked_loss = loss_fct(masked_logits, masked_labels)
            criterion = nn.MSELoss()
            total_loss = 0.0
            regression_labels = regression_labels.transpose(0,1)
            for i in range(553):
                predicted = regression_outputs[i]
                true_value = regression_labels[i]
                loss = criterion(predicted.reshape(-1), true_value)
                total_loss += loss
            Total_loss = masked_loss+ total_loss/553
            return Total_loss, regression_outputs, total_loss/553
        else:
            return regression_outputs
        
        
        #if masked_labels is not None and labels is not None:
            #masked_positions, masked_values = zip(*masked_labels)
            #logits = logits[masked_positions]  # 仅保留masked位置的预测值
            #masked_values = torch.tensor(masked_values).to(logits.device)
            #loss_fct = nn.CrossEntropyLoss()
            #masked_loss = loss_fct(logits, masked_values)
            #regression_loss = nn.functional.mse_loss(regression_output.view(-1), labels.view(-1))
            #simcse_loss = self.calculate_simcse_loss(simcse_embeddings1,simcse_embeddings2)
            #Total_loss = masked_loss+ regression_loss + simcse_loss
            #return Total_loss, logits, regression_output
        #else:
            #return logits, regression_output
        
        #if labels is not None:
            #loss_fct = nn.CrossEntropyLoss()
            #classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #simcse_loss = self.calculate_simcse_loss(simcse_embeddings1,simcse_embeddings2)
            #Total_loss = classification_loss + simcse_loss
            #return Total_loss, logits
        #else:
            #return logits
