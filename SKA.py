import hyperbolic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trm

from MHA import MHA


class SKA(nn.Module):
    def __init__(self, args, num_entities, num_rels, num_types, pretrained_embs, curvature=1):
        super(SKA, self).__init__()
        self.num_entities = num_entities
        self.num_rels = num_rels
        self.num_types = num_types
        self.dataset = args['dataset']
        self.use_cuda = args['cuda']
        self.device = torch.device('cuda')
        self.emb_dim = args['emb_dim']
        self.entity_embs = self._init_entity_embedding(num_entities, dim=self.emb_dim)
        self.relation_embs = self._init_relation_embedding(num_rels, dim=self.emb_dim)
        self.type_embs = self._init_entity_embedding(num_types, dim=self.emb_dim)
        self.pretrained_ent_embs = pretrained_embs[0]
        self.pretrained_rel_embs = pretrained_embs[1]
        self.pretrained_type_embs = pretrained_embs[2]
        if 'bert-base-uncased' == args['plm']:
            self.embedding_dim = 768
        elif 'bert-large-uncased' == args['plm']:
            self.embedding_dim = 1024
        elif 'roberta-base' == args['plm']:
            self.embedding_dim = 768
        elif 'roberta-large' == args['plm']:
            self.embedding_dim = 1024
        elif 'gpt2' == args['plm']:
            self.embedding_dim = 768
        else:
            self.embedding_dim = 768

        self.mlp_t = MLP(self.embedding_dim, self.emb_dim)
        self.mlp_s = MLP(self.emb_dim, self.emb_dim)
        # self.kg_decoder = nn.Linear(self.emb_dim, self.num_types)
        # self.et_decoder = nn.Linear(self.emb_dim, self.num_types)
        self.decoder = nn.Linear(self.emb_dim, self.num_types)
        self.num_ent_neighbors = args['sample_kg_size']
        self.num_et_neighbors = args['sample_et_size']
        self.temperature = args['temperature']
        self.mha = MHA(5, 1.0)
        self.curvature = curvature
        self.linear = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    def _init_entity_embedding(self, num_entities, dim=100):
        """
        entity embedding: uniform(-6/sqrt(dim), 6/sqrt(dim)
        :return: matrix of entity embeddings
        """
        entity_embeddings = nn.Embedding(num_embeddings=num_entities,
                                         embedding_dim=dim, )
        uniform_range = np.sqrt(6) / np.sqrt(num_entities + dim)
        entity_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        return entity_embeddings

    def _init_relation_embedding(self, num_relations, dim=100):
        """
        relation embedding: l = uniform(-6/sqrt(dim), 6/sqrt(dim))
        l = l / ||l||
        :return: matrix of relation embeddings
        """
        relation_embeddings = nn.Embedding(num_embeddings=num_relations,
                                           embedding_dim=dim)
        uniform_range = np.sqrt(6) / np.sqrt(num_relations + dim)
        relation_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        # relation_embeddings.weight.data[:, :].div_(relation_embeddings.weight.data[:, :].norm(p=1, dim=1, keepdim=True))
        relation_embeddings.weight.data[:, :].div_(relation_embeddings.weight.data[:, :].norm(p=2, dim=1, keepdim=True))
        return relation_embeddings

    def forward(self, et_content, kg_content, two_hop_et_content=None, two_hop_kg_content=None, two_hop_calc_mask=None,
                three_hop_et_content=None, three_hop_kg_content=None, three_hop_calc_mask=None):
        batch_size, et_neighbor_size = et_content[:, :, 2].size()
        et_type_ids = et_content[:, :, 2].view(-1) - self.num_entities
        et_type_structural_embs = self.type_embs(et_type_ids)
        #   et_type_structural_embs = torch.index_select(self.type_embs, 0, et_type_ids)
        et_type_ids = et_type_ids.cpu()
        et_type_textural_embs = torch.index_select(self.pretrained_type_embs, 0, et_type_ids)
        et_type_textural_embs = et_type_textural_embs.to(et_type_structural_embs.device)
        #   整合type textual embeddings 和 type structural embeddings
        et_type_structural_embs = self.mlp_s(et_type_structural_embs)
        et_type_textural_embs = self.mlp_t(et_type_textural_embs)
        #   这里我们要测试是否需要对textual embeddings进行l2正则化处理
        #   Justin Lovelance的文章有提出正则化有利于解决BERT embeddings的分布不均匀问题。
        et_type_embs = et_type_textural_embs + et_type_structural_embs

        _, kg_neighbor_size = kg_content[:, :, 2].size()
        kg_entity_ids = kg_content[:, :, 2].view(-1)
        #   kg_entity_structural_embs = torch.index_select(self.entity_embs, 0, kg_entity_ids)
        kg_entity_structural_embs = self.entity_embs(kg_entity_ids)
        kg_entity_ids = kg_entity_ids.cpu()
        kg_entity_textural_embs = torch.index_select(self.pretrained_ent_embs, 0, kg_entity_ids)
        kg_entity_textural_embs = kg_entity_textural_embs.to(kg_entity_structural_embs.device)
        #   整合entity textual embeddings 和 entity structural embeddings
        kg_entity_structural_embs = self.mlp_s(kg_entity_structural_embs)
        kg_entity_textural_embs = self.mlp_t(kg_entity_textural_embs)
        kg_entity_embs = kg_entity_textural_embs + kg_entity_structural_embs

        #   整合relation textual embeddings 和 relation structural embeddings
        kg_relation_ids = kg_content[:, :, 1].view(-1)
        #   kg_relation_structural_embs = torch.index_select(self.relation_embs, 0, kg_relation_ids % self.num_rels)
        kg_relation_structural_embs = self.relation_embs(kg_relation_ids % self.num_rels)
        kg_relation_ids = kg_relation_ids.cpu()
        kg_relation_textural_embs = torch.index_select(self.pretrained_rel_embs, 0, kg_relation_ids % self.num_rels)
        kg_relation_textural_embs = kg_relation_textural_embs.to(kg_relation_structural_embs.device)
        kg_relation_ids = kg_relation_ids.to(kg_entity_structural_embs.device)
        kg_relation_structural_embs = self.mlp_s(kg_relation_structural_embs)
        kg_relation_textural_embs = self.mlp_t(kg_relation_textural_embs)
        kg_relation_embs = kg_relation_textural_embs + kg_relation_structural_embs

        #   message passing
        kg_relation_embs[kg_relation_ids >= self.num_rels] = kg_relation_embs[kg_relation_ids >= self.num_rels] * -1
        #   这里先尝试TransE
        translated_embs = kg_entity_embs + kg_relation_embs
        translated_embs = translated_embs.reshape(batch_size, -1, self.emb_dim)
        et_type_embs = et_type_embs.reshape(batch_size, -1, self.emb_dim)
        #   用hyperbolic加法
        # kg_entity_embs_proj = hyperbolic.project(kg_entity_embs, self.curvature)
        # kg_relation_embs_proj = hyperbolic.project(kg_relation_embs, self.curvature)
        # translated_embs = hyperbolic.mobius_add(kg_entity_embs_proj, kg_relation_embs_proj, self.curvature)
        # translated_embs = hyperbolic.project(translated_embs, self.curvature)
        # translated_embs = translated_embs.reshape(batch_size, -1, self.emb_dim)
        #   hyperbolic 空间转换
        # et_type_embs_proj = hyperbolic.project(et_type_embs, self.curvature)
        # # et_type_embs_proj = hyperbolic.expmap0(self.linear(hyperbolic.logmap0(et_type_embs_proj, self.curvature)),
        # #                                        self.curvature)
        # # et_type_embs_proj = hyperbolic.project(et_type_embs_proj, self.curvature)
        # et_type_embs = et_type_embs_proj.reshape(batch_size, -1, self.emb_dim)

        # kg_predict = self.kg_decoder(translated_embs)
        # et_predict = self.et_decoder(et_type_embs)
        kg_predict = self.decoder(translated_embs)
        et_predict = self.decoder(et_type_embs)
        all_embs = torch.cat([translated_embs, et_type_embs], dim=1)
        global_emb = torch.mean(all_embs, dim=1, keepdim=True)
        # global_predict = self.decoder(all_embs)
        global_predict = self.decoder(global_emb)

        if two_hop_et_content is not None:
            # one_hop_neighbor_ent = kg_content[:, :, 2]
            # one_hop_neighbor_rel = kg_content[:, :, 1]
            #   kg_relation_embs其實就是這些rel的emb，只是沒有reshape

            neighbor_type_ids = two_hop_et_content[:, :, :, 2].view(-1) - self.num_entities
            neighbor_type_structural_embs = self.type_embs(neighbor_type_ids)
            neighbor_type_ids = neighbor_type_ids.cpu()
            neighbor_type_textual_embs = torch.index_select(self.pretrained_type_embs, 0, neighbor_type_ids)
            neighbor_type_textual_embs = neighbor_type_textual_embs.to(neighbor_type_structural_embs.device)
            neighbor_type_structural_embs = self.mlp_s(neighbor_type_structural_embs)
            neighbor_type_textual_embs = self.mlp_t(neighbor_type_textual_embs)
            neighbor_type_embs = neighbor_type_textual_embs + neighbor_type_structural_embs

            #   hyperbolic 空间转换
            # neighbor_type_embs_proj = hyperbolic.project(neighbor_type_embs, self.curvature)
            # neighbor_type_embs_proj = hyperbolic.expmap0(self.linear(hyperbolic.logmap0(
            #     neighbor_type_embs_proj, self.curvature)), self.curvature)
            # neighbor_type_embs_proj = hyperbolic.project(et_type_embs_proj, self.curvature)

            two_hop_neighbor_ent = two_hop_kg_content[:, :, :, 2].view(-1)
            two_hop_neighbor_ent_structural_embs = self.entity_embs(two_hop_neighbor_ent)
            two_hop_neighbor_ent = two_hop_neighbor_ent.cpu()
            two_hop_neighbor_ent_textual_embs = torch.index_select(self.pretrained_ent_embs, 0, two_hop_neighbor_ent)
            two_hop_neighbor_ent_textual_embs = two_hop_neighbor_ent_textual_embs.to(two_hop_neighbor_ent_structural_embs.device)
            two_hop_neighbor_ent_structural_embs = self.mlp_s(two_hop_neighbor_ent_structural_embs)
            two_hop_neighbor_ent_textual_embs = self.mlp_t(two_hop_neighbor_ent_textual_embs)
            two_hop_neighbor_ent_embs = two_hop_neighbor_ent_textual_embs + two_hop_neighbor_ent_structural_embs

            two_hop_rel = two_hop_kg_content[:, :, :, 1].view(-1)
            two_hop_rel_structural_embs = self.relation_embs(two_hop_rel % self.num_rels)
            two_hop_rel = two_hop_rel.cpu()
            two_hop_rel_textual_embs = torch.index_select(self.pretrained_rel_embs, 0, two_hop_rel % self.num_rels)

            two_hop_rel_textual_embs = two_hop_rel_textual_embs.to(two_hop_rel_structural_embs.device)
            two_hop_rel = two_hop_rel.to(two_hop_rel_structural_embs.device)
            two_hop_rel_structural_embs = self.mlp_s(two_hop_rel_structural_embs)
            two_hop_rel_textual_embs = self.mlp_t(two_hop_rel_textual_embs)
            two_hop_rel_embs = two_hop_rel_textual_embs + two_hop_rel_structural_embs
            #   2hop message passing
            two_hop_rel_embs[two_hop_rel > self.num_rels] = two_hop_rel_embs[two_hop_rel > self.num_rels] * -1
            #   先用TransE操作
            two_hop_translated_embs = two_hop_neighbor_ent_embs + two_hop_rel_embs
            two_hop_translated_embs = two_hop_translated_embs.reshape(two_hop_kg_content.shape[0],
                                                                      two_hop_kg_content.shape[1],
                                                                      two_hop_kg_content.shape[2], -1)

            neighbor_type_embs = neighbor_type_embs.reshape(two_hop_kg_content.shape[0],
                                                            two_hop_kg_content.shape[1],
                                                            two_hop_et_content.shape[2], -1)
            one_hop_neighbor_agg_embs = torch.cat([two_hop_translated_embs, neighbor_type_embs], dim=-2)


            #   使用hyperbolic加法
            # two_hop_neighbor_ent_embs_proj = hyperbolic.project(two_hop_neighbor_ent_embs, self.curvature)
            # two_hop_rel_embs_proj = hyperbolic.project(two_hop_rel_embs, self.curvature)
            # two_hop_translated_embs = hyperbolic.mobius_add(two_hop_neighbor_ent_embs_proj, two_hop_rel_embs_proj,
            #                                                 self.curvature)
            # two_hop_translated_embs_proj = hyperbolic.project(two_hop_translated_embs, self.curvature)
            #
            # neighbor_type_embs_proj = neighbor_type_embs_proj.reshape(two_hop_kg_content.shape[0],
            #                                                           two_hop_kg_content.shape[1],
            #                                                           two_hop_et_content.shape[2], -1)
            # two_hop_translated_embs_proj = two_hop_translated_embs_proj.reshape(two_hop_kg_content.shape[0],
            #                                                                two_hop_kg_content.shape[1],
            #                                                                two_hop_kg_content.shape[2], -1)
            #
            # one_hop_neighbor_agg_embs = torch.cat([two_hop_translated_embs_proj, neighbor_type_embs_proj], dim=-2)

            one_hop_neighbor_agg_embs = two_hop_calc_mask.unsqueeze(-1).unsqueeze(-1) * one_hop_neighbor_agg_embs
            one_hop_neighbor_agg_embs = torch.mean(one_hop_neighbor_agg_embs, dim=-2)
            #   先尝试translation
            one_hop_neighbor_agg_embs = F.normalize(one_hop_neighbor_agg_embs, dim=-1)
            two_hop_agg_embs = one_hop_neighbor_agg_embs + kg_relation_embs.reshape(batch_size, kg_neighbor_size, -1)
            #   使用hyperbolic
            # one_hop_neighbor_agg_embs = hyperbolic.project(one_hop_neighbor_agg_embs, self.curvature)
            # one_hop_neighbor_agg_embs = one_hop_neighbor_agg_embs.reshape(batch_size * kg_neighbor_size, -1)
            # two_hop_agg_embs = hyperbolic.mobius_add(one_hop_neighbor_agg_embs, kg_relation_embs_proj, self.curvature)
            # two_hop_agg_embs = hyperbolic.project(two_hop_agg_embs, self.curvature)

            two_hop_agg_embs = two_hop_agg_embs.reshape(batch_size, kg_neighbor_size, -1)
            two_hop_agg_predict = self.decoder(two_hop_agg_embs)
            two_hop_agg_global_emb = torch.mean(two_hop_agg_embs, dim=1, keepdim=True)
            two_hop_agg_global_predict = self.decoder(two_hop_agg_global_emb)
        if three_hop_et_content is not None:
            two_hop_neighbor_type_ids = three_hop_et_content[:, :, :, 2].view(-1) - self.num_entities
            two_hop_neighbor_type_structural_embs = self.type_embs(two_hop_neighbor_type_ids)
            two_hop_neighbor_type_ids = two_hop_neighbor_type_ids.cpu()
            two_hop_neighbor_type_textual_embs = torch.index_select(self.pretrained_type_embs, 0, two_hop_neighbor_type_ids)
            two_hop_neighbor_type_textual_embs = two_hop_neighbor_type_textual_embs.to(two_hop_neighbor_type_structural_embs.device)
            two_hop_neighbor_type_structural_embs = self.mlp_s(two_hop_neighbor_type_structural_embs)
            two_hop_neighbor_type_textual_embs = self.mlp_t(two_hop_neighbor_type_textual_embs)
            two_hop_neighbor_type_embs = two_hop_neighbor_type_textual_embs + two_hop_neighbor_type_structural_embs

            three_hop_neighbor_ent = three_hop_kg_content[:, :, :, 2].view(-1)
            three_hop_neighbor_ent_structural_embs = self.entity_embs(three_hop_neighbor_ent)
            three_hop_neighbor_ent = three_hop_neighbor_ent.cpu()
            three_hop_neighbor_ent_textual_embs = torch.index_select(self.pretrained_ent_embs, 0, three_hop_neighbor_ent)
            three_hop_neighbor_ent_textual_embs = three_hop_neighbor_ent_textual_embs.to(
                three_hop_neighbor_ent_structural_embs.device)
            three_hop_neighbor_ent_structural_embs = self.mlp_s(three_hop_neighbor_ent_structural_embs)
            three_hop_neighbor_ent_textual_embs = self.mlp_t(three_hop_neighbor_ent_textual_embs)
            three_hop_neighbor_ent_embs = three_hop_neighbor_ent_textual_embs + three_hop_neighbor_ent_structural_embs

            three_hop_rel = three_hop_kg_content[:, :, :, 1].view(-1)
            three_hop_rel_structural_embs = self.relation_embs(three_hop_rel % self.num_rels)
            three_hop_rel = three_hop_rel.cpu()
            three_hop_rel_textual_embs = torch.index_select(self.pretrained_rel_embs, 0, three_hop_rel % self.num_rels)
            three_hop_rel_textual_embs = three_hop_rel_textual_embs.to(three_hop_rel_structural_embs.device)
            three_hop_rel = three_hop_rel.to(three_hop_rel_structural_embs.device)
            three_hop_rel_structural_embs = self.mlp_s(three_hop_rel_structural_embs)
            three_hop_rel_textual_embs = self.mlp_t(three_hop_rel_textual_embs)
            three_hop_rel_embs = three_hop_rel_textual_embs + three_hop_rel_structural_embs
            #   3hop message passing
            three_hop_rel_embs[three_hop_rel > self.num_rels] = three_hop_rel_embs[three_hop_rel > self.num_rels] * -1
            three_hop_translated_embs = three_hop_neighbor_ent_embs + three_hop_rel_embs
            three_hop_translated_embs = three_hop_translated_embs.reshape(three_hop_kg_content.shape[0],
                                                                          three_hop_kg_content.shape[1],
                                                                          three_hop_kg_content.shape[2], -1)
            two_hop_neighbor_type_embs = two_hop_neighbor_type_embs.reshape(three_hop_kg_content.shape[0],
                                                                            three_hop_kg_content.shape[1],
                                                                            three_hop_et_content.shape[2], -1)
            two_hop_neighbor_agg_embs = torch.cat([three_hop_translated_embs, two_hop_neighbor_type_embs], dim=-2)

            two_hop_neighbor_agg_embs = three_hop_calc_mask.unsqueeze(-1).unsqueeze(-1) * two_hop_neighbor_agg_embs
            two_hop_neighbor_agg_embs = torch.mean(two_hop_neighbor_agg_embs, -2)
            two_hop_neighbor_agg_embs = F.normalize(two_hop_neighbor_agg_embs, dim=-1)

            #   two_hop_neighbor_agg_embs 代替的應該是第159行的two_hop_neighbor_ent_embs
            two_hop_agg_translated_embs = two_hop_neighbor_agg_embs.reshape(
                -1, two_hop_neighbor_agg_embs.shape[-1]) + two_hop_rel_embs
            two_hop_agg_translated_embs = two_hop_agg_translated_embs.reshape(two_hop_kg_content.shape[0],
                                                                              two_hop_kg_content.shape[1],
                                                                              two_hop_kg_content.shape[2], -1)
            one_hop_neighbor_agg3_embs = torch.cat([two_hop_agg_translated_embs, neighbor_type_embs], dim=-2)
            one_hop_neighbor_agg3_embs = two_hop_calc_mask.unsqueeze(-1).unsqueeze(-1) * one_hop_neighbor_agg3_embs
            one_hop_neighbor_agg3_embs = torch.mean(one_hop_neighbor_agg3_embs, dim=-2)
            one_hop_neighbor_agg3_embs = F.normalize(one_hop_neighbor_agg3_embs, dim=-1)
            three_hop_agg_embs = one_hop_neighbor_agg3_embs + kg_relation_embs.reshape(batch_size, kg_neighbor_size, -1)
            three_hop_agg_embs = three_hop_agg_embs.reshape(batch_size, kg_neighbor_size, -1)
            three_hop_agg_predict = self.decoder(three_hop_agg_embs)
            three_hop_agg_global_emb = torch.mean(three_hop_agg_embs, dim=1, keepdim=True)
            three_hop_agg_global_predict = self.decoder(three_hop_agg_global_emb)
            predict = torch.cat([kg_predict, et_predict, global_predict, two_hop_agg_predict,
                                 two_hop_agg_global_predict, three_hop_agg_predict,
                                 three_hop_agg_global_predict], dim=1)
        elif two_hop_et_content is not None:
            predict = torch.cat([kg_predict, et_predict, global_predict, two_hop_agg_predict,
                                 two_hop_agg_global_predict], dim=1)
        else:
            predict = torch.cat([kg_predict, et_predict, global_predict], dim=1)
        predict = self.mha(predict).sigmoid()
        return predict


class MLP(nn.Module):
    def __init__(self, dim1, dim2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim1, int(dim1 / 2))
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(int(dim1 / 2), dim2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        #   x = self.tanh(x)
        return F.normalize(x, dim=-1)

