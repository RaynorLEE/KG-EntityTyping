import torch
import torch.nn as nn

from transformers import BertConfig, BertModel

device = torch.device('cuda:0')

class PLM(nn.Module):
    def __init__(self, plm='bert-base-uncased', pretrained_model=None):
        super(PLM, self).__init__()
        self.lm_config = BertConfig.from_pretrained('bert-base-uncased')
        self.lm_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.use_pretrained_model = False
        if pretrained_model is not None:
            self.use_pretrained_model = True
            self.lm_encoder = pretrained_model.lm_encoder
            self.lm_encoder.pooler = pretrained_model.lm_pooler

    def forward(self, seq_tokens):
        outputs = self.lm_encoder(**seq_tokens)
        pooler_outputs = outputs['pooler_output']
        if self.use_pretrained_model is True:
            return pooler_outputs[:, 0, :]
        else:
            return pooler_outputs


def pretrain_representations(semantics, plm, tokenizer):
    with torch.no_grad():
        start = 0
        bs = 32
        total = len(semantics)
        end = 32
        batch_semantics = semantics[start:end].tolist()
        batch_seq_tokens = tokenizer(batch_semantics, padding=True, truncation=True, return_tensors='pt')
        batch_seq_tokens = batch_seq_tokens.to(device)
        pooler_outputs = plm(batch_seq_tokens)
        embeddings = pooler_outputs.cpu()

        start = 32
        end = start + bs
        while start < total:
            if end > total:
                end = total
            batch_semantics = semantics[start:end].tolist()
            batch_seq_tokens = tokenizer(batch_semantics, padding=True, truncation=True, return_tensors='pt')
            batch_seq_tokens = batch_seq_tokens.to(device)
            pooler_outputs = plm(batch_seq_tokens)
            batch_embeddings = pooler_outputs.cpu()
            embeddings = torch.cat([embeddings, batch_embeddings], dim=0)
            start = start + bs
            end = start + bs
        return embeddings
