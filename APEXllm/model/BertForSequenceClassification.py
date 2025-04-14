from transformers import BertModel
import torch.nn as nn
import torch
import esm

class SequenceClassifier(nn.Module):
    def __init__(self, config,hidden_size):
        super(SequenceClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.dense = nn.Linear(hidden_size, config.num_labels)
        
    def forward(self, hidden_states):
        hidden_states = self.dense(self.dropout(hidden_states))
        return hidden_states
#Define the SimCSE contrastive loss, similar to the previous example.
class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SimCSELoss, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, embeddings1, embeddings2):
        embeddings1 = nn.functional.normalize(self.dropout(embeddings1), dim=-1, p=2)
        embeddings2 = nn.functional.normalize(self.dropout(embeddings2), dim=-1, p=2)
        cosine_similarity = nn.functional.cosine_similarity(embeddings1, embeddings2)
        #similarity_matrix = torch.matmul(embeddings1, embeddings2.transpose(1, 2))
        #logits = similarity_matrix / self.temperature
        #batch_size = logits.size(0)
        #labels = torch.arange(logits.size(0),).to(logits.device)
        #loss = nn.functional.cross_entropy(logits.view(-1, batch_size), labels)
        return 1-cosine_similarity.mean()*self.temperature
'''

class MLPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1280
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x
class Similarity(nn.Module):
        """
        Dot product or cosine similarity
        """

        def __init__(self, temp):
            super().__init__()
            self.temp = temp
            self.cos = nn.CosineSimilarity(dim=-1)

        def forward(self, x, y):
            return self.cos(x, y) / self.temp
class SimCSELoss(nn.Module):
    def __init__(self, config):
        super(SimCSELoss, self).__init__()
        self.temperature = 0.05
        self.dropout = nn.Dropout(p=0.1)
        self.mlp = MLPLayer()
        self.sim = Similarity(self.temperature)
        self.device = config.device
    def forward(self, embeddings1, embeddings2):
        embeddings1 = self.mlp(embeddings1)
        embeddings2 = self.mlp(embeddings2)
        cos_sim = self.sim(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        return loss
'''
class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = config.num_labels
        self.bert,self.tokenize = esm.pretrained.esm2_t48_15B_UR50D()
        del self.bert.contact_head
        hidden_size = 5120
        self.linear = torch.nn.Linear(hidden_size, 1)   
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = config.device
        self.classifier = SequenceClassifier(config,hidden_size)
        #self.calculate_simcse_loss = SimCSELoss(temperature=0.05)
    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):
        """

        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        output = self.bert(input_ids,repr_layers=[48],return_contacts=False)  # [batch_size,hidden_size]
        #output2 = self.bert(input_ids,repr_layers=[36],return_contacts=False)
        pooled_output = output ['representations'][48][:, :-1, :]  # [batch_size, hidden_size]
        padding_mask = torch.where(input_ids[:, :-1] == 1, 0, 1)
        x = self.linear(pooled_output)
        x = self.softmax(x)
        x = x * padding_mask.unsqueeze(-1)
        pooled_output = pooled_output * x
        pooled_output = pooled_output.sum(dim=1)
        # pooled_output = self.dropout(pooled_output)
        #simcse_embeddings1 = output['representations'][33][:, 0, :]
        # Embeddings for the second time
        #simcse_embeddings2 = output2['representations'][33][:, 0, :]
        logits = self.classifier(pooled_output)     # [batch_size, num_label]
        if labels is not None:
            class_weights = torch.tensor([0.1, 0.9]).to(self.device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)

            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #simcse_loss = self.calculate_simcse_loss(simcse_embeddings1,simcse_embeddings2)
            #Total_loss = classification_loss + simcse_loss
            Total_loss = classification_loss
            return Total_loss, logits
        else:
            return logits
