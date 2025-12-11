import torch
import torch.nn as nn
from transformers import DistilBertModel

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(768, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_embedding)
        probs = self.softmax(logits)
        return probs

