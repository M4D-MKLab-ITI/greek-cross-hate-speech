from transformers import BertModel, AutoModel
import torch 
from torch import nn 
import sklearn
import math 

def loss_fn(logits,labels):
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels.view(-1))
    return loss

class HateModel(nn.Module): 
    def __init__(self, lm='bert-base-uncased', config=None) -> None:
        super().__init__()
        
        self.config = config
        self.lm = AutoModel.from_pretrained(lm , cache_dir='/home/nstylia/projects/greek-hatespeech/hfcache')
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None): 
        output = self.lm(input_ids, attention_mask=attention_mask)
        logits = self.classifier(output['pooler_output'])

        loss = None
        if labels!=None: 
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels)
        return loss, logits