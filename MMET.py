import torch
import torch.nn as nn
import trm

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from MHA import MHA
from transformers import BertConfig, BertModel


class MMET(nn.Module):
    def __init__(self, args, num_entities, num_rels, num_types):
        super(MMET, self).__init__()
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.num_types = num_types
        self.dataset = args['dataset']
        self.use_cuda = args['cuda']
        self.num_queries = args['nquery']
        self.device = torch.device('cuda')

        self.num_triples_sampled = args['sample_kg_size']
        self.num_types_sampled = args['sample_et_size']

        self.lm_config = BertConfig.from_pretrained('bert-base-uncased')
        self.lm_encoder = BertModel.from_pretrained('bert-base-uncased')

        # self.lm_pooler = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.Tanh(),
        # )

        self.lm_pooler = nn.Sequential(
            self.lm_encoder.pooler.dense,
            nn.Tanh(),
        )

        self.decoder = nn.Linear(768, num_types)
        self.relu = nn.ReLU()

        self.num_ent_neighbors = args['sample_kg_size']
        self.num_et_neighbors = args['sample_et_size']
        self.temperature = args['temperature']
        self.mha = MHA(5, 1.0)

        self.query_emb = nn.Parameter(torch.empty([self.num_queries, 768]))
        nn.init.xavier_uniform_(self.query_emb)

        # self.contextual_encoder = BertModel.from_pretrained('bert-base-uncased')
        contextual_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.contextual_encoder = nn.TransformerEncoder(contextual_encoder_layer, num_layers=3)

        self.queue_length = 1280
        self.register_buffer("queue", torch.randn(768, self.queue_length))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.moco_temp = 0.07
        self.queue_ptr = 0
        self.warm_up = True
        #self.layer = MMETLayer(args, self.embedding_dim, num_types, args['temperature'])

    def forward(self, kg_seq_tokens, kg_mask_index, et_seq_tokens, et_mask_index, ent_seq_tokens, bs, mode='train'):
        # prediction = self.lm_encoder(kg_seq_tokens)['last_hidden_state'][]
        #   ATTENTION: The batch size CANNOT be larger than 48 since RTX3090 does not have enough GPU memory for larger
        #   batch size
        kg_outputs = self.lm_encoder(**kg_seq_tokens)
        kg_masked_token = kg_outputs['last_hidden_state'][kg_mask_index[0], kg_mask_index[1]]
        kg_pooled_output = self.lm_pooler(kg_masked_token)
        #   kg_pooled_output = kg_outputs.pooler_output

        et_outputs = self.lm_encoder(**et_seq_tokens)
        et_masked_token = et_outputs['last_hidden_state'][et_mask_index[0], et_mask_index[1]]
        et_pooled_output = self.lm_pooler(et_masked_token)
        #   et_pooled_output = et_outputs.pooler_output

        num_ent_neighbors = int(kg_pooled_output.shape[0] / bs)
        kg_pooled_output = kg_pooled_output.reshape(bs, num_ent_neighbors, -1)
        kg_predict = self.decoder(kg_pooled_output)
        # num_ent_neighbors = int(kg_predict.shape[0] / bs)
        # kg_predict = kg_predict.reshape(bs, num_ent_neighbors, -1)

        num_et_neighbors = int(et_pooled_output.shape[0] / bs)
        et_pooled_output = et_pooled_output.reshape(bs, num_et_neighbors, -1)
        et_predict = self.decoder(et_pooled_output)
        # num_et_neighbors = int(et_predict.shape[0] / bs)
        # et_predict = et_predict.reshape(bs, num_et_neighbors, -1)

        all_pooled_output = torch.cat([kg_pooled_output, et_pooled_output], dim=1)
        all_pooled_output = all_pooled_output.mean(dim=1, keepdim=True)
        global_predict = self.decoder(all_pooled_output)

        #   这里我先把ent textual embedding 当作query，把structural encoding当作key
        if mode == 'train':
            with torch.no_grad():
                ent_outputs = self.lm_encoder(**ent_seq_tokens)
                ent_cls_token = ent_outputs['last_hidden_state'][:, 0]

                ent_pooled_output = self.lm_pooler(ent_cls_token)
                all_pooled_output = all_pooled_output.squeeze(1)
                q = nn.functional.normalize(ent_pooled_output, dim=1)
            k = nn.functional.normalize(all_pooled_output, dim=1)
            #   computing logits
            #   positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            #   negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            moco_logits = torch.cat([l_pos, l_neg], dim=1)
            moco_logits /= self.moco_temp
            moco_labels = torch.zeros(moco_logits.shape[0], dtype=torch.long).cuda()
            self._dequeue_and_enqueue(k)

        # batch_query_emb = self.query_emb.repeat(bs, 1, 1)
        # contextual_enc_input = torch.cat([batch_query_emb, kg_masked_token.reshape(bs, -1, 768), et_masked_token.reshape(bs, -1, 768)], dim=1)
        # contextual_token_outputs = self.contextual_encoder(contextual_enc_input)[:,0:self.num_queries]
        # contextual_pooled_output = self.lm_pooler(contextual_token_outputs)
        # contextual_predict = self.decoder(contextual_pooled_output)

        #   predict = torch.cat([kg_predict, et_predict], dim=1)
        predict = torch.cat([kg_predict, et_predict, global_predict], dim=1)
        #   predict = torch.cat([kg_predict, et_predict, contextual_predict, global_predict], dim=1)
        #   weight = torch.softmax(self.temperature * predict, dim=1)
        #   predict = (predict * weight.detach()).sum(1).sigmoid()
        predict = self.mha(predict).sigmoid()
        if mode == 'train':
            return predict, moco_logits, moco_labels
        else:
            return predict

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #   keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        #   assert self.queue_length % batch_size == 0  # for simplicity
        if ptr + batch_size >= self.queue_length:
            #   预热结束
            self.warm_up = False
        if ptr + batch_size > self.queue_length:
            rear_num = self.queue_length - ptr
            front_num = ptr + batch_size - self.queue_length
            self.queue[:, ptr:] = keys[0:rear_num].T
            self.queue[:, 0:front_num] = keys[rear_num:].T
            ptr = front_num
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_length  # move pointer
        self.queue_ptr = ptr

# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output


# class MMETLayer(nn.Module):
#     def __init__(self, args, embedding_dim, num_types, temperature):
#         super(MMETLayer, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_types = num_types
#         self.fc = nn.Linear(embedding_dim, num_types)
#         self.temperature = temperature
#         self.device = torch.device('cuda')
#
#         self.trm_nlayer = args['trm_nlayer']
#         self.trm_nhead = args['trm_nhead']
#         self.trm_hidden_dropout = args['trm_hidden_dropout']
#         self.trm_attn_dropout = args['trm_attn_dropout']
#         self.trm_ff_dim = args['trm_ff_dim']
#         self.global_pos_size = args['global_pos_size']
#         self.embedding_range = 10 / self.embedding_dim
#
#         self.global_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim))
#         torch.nn.init.normal_(self.global_cls, std=self.embedding_range)
#         self.pos_embeds = nn.Embedding(self.global_pos_size, self.embedding_dim)
#         torch.nn.init.normal_(self.pos_embeds.weight, std=self.embedding_range)
#         self.layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)
#
#         self.transformer_encoder = trm.Encoder(
#             lambda: trm.EncoderLayer(
#                 self.embedding_dim,
#                 trm.MultiHeadedAttentionWithRelations(
#                     self.trm_nhead,
#                     self.embedding_dim,
#                     self.trm_attn_dropout),
#                 trm.PositionwiseFeedForward(
#                     self.embedding_dim,
#                     self.trm_ff_dim,
#                     self.trm_hidden_dropout),
#                 num_relation_kinds=0,
#                 dropout=self.trm_hidden_dropout),
#             self.trm_nlayer,
#             self.embedding_range,
#             tie_layers=False)
#
#     def convert_mask_trm(self, attention_mask):
#         attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
#         return attention_mask
#
#     def forward(self, local_embedding, global_embedding):
#         local_msg = torch.relu(local_embedding)
#         predict1 = self.fc(local_msg)
#
#         batch_size, neighbor_size, emb_size = local_embedding.size()
#         attention_mask = torch.ones(batch_size, neighbor_size + 1).bool().to(self.device)
#         second_local = torch.cat([self.global_cls.expand(batch_size, 1, emb_size), local_embedding], dim=1)
#         pos = self.pos_embeds(torch.arange(0, 3).to(self.device))
#         second_local[:, 0] = second_local[:, 0] + pos[0].unsqueeze(0)
#         second_local[:, 1] = second_local[:, 1] + pos[1].unsqueeze(0)
#         second_local[:, 2:] = second_local[:, 2:] + pos[2].view(1, 1, -1)
#         second_local = self.layer_norm(second_local)
#         second_local = self.transformer_encoder(second_local, None, self.convert_mask_trm(attention_mask))
#         second_local = second_local[-1][:, :2][:, 0].unsqueeze(1)
#         predict2 = self.fc(torch.relu(second_local))
#         predict3 = self.fc(torch.relu(global_embedding))
#
#         predict = torch.cat([predict1, predict2, predict3], dim=1)
#         weight = torch.softmax(self.temperature * predict, dim=1)
#         predict = (predict * weight.detach()).sum(1).sigmoid()
#
#         return predict
#
