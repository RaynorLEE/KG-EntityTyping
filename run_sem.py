import argparse
from utils import *
# from TET import TET
from SEM import SEM
from dataloader import SEMdataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

device = torch.device('cuda:0')

def main(args):
    use_cuda = args['cuda'] and torch.cuda.is_available()
    data_path = os.path.join(args['data_dir'], args['dataset'])
    save_path = os.path.join(args['save_dir'], args['save_path'])

    e2id = read_id(os.path.join(data_path, 'entities.tsv'))
    r2id = read_id(os.path.join(data_path, 'relations.tsv'))
    t2id = read_id(os.path.join(data_path, 'types.tsv'))
    c2id = read_id(os.path.join(data_path, 'clusters.tsv'))
    e2desc, e2text = read_entity_wiki(os.path.join(data_path, 'entity_wiki.json'), e2id, args['semantic'])
    r2text = read_rel_context(os.path.join(data_path, 'relation2text.txt'), r2id)
    t2desc = read_type_context(os.path.join(data_path, 'hier_type_desc.txt'), t2id)
    num_entities = len(e2id)
    num_rels = len(r2id)
    num_types = len(t2id)
    num_clusters = len(c2id)
    train_type_label, test_type_label = load_train_all_labels(data_path, e2id, t2id)
    if use_cuda:
        sample_ent2pair = torch.LongTensor(load_entity_cluster_type_pair_context(args, r2id, e2id)).cuda()
    train_dataset = SEMdataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
    valid_dataset = SEMdataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')
    test_dataset = SEMdataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args['train_batch_size'],
                                  shuffle=True,
                                  collate_fn=SEMdataset.collate_fn,
                                  num_workers=6)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args['train_batch_size'],
                                  shuffle=False,
                                  collate_fn=SEMdataset.collate_fn,
                                  num_workers=6)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args['test_batch_size'],
                                 shuffle=False,
                                 collate_fn=SEMdataset.collate_fn,
                                 num_workers=6)

    # model = TET(args, num_entities, num_rels, num_types, num_clusters)
    model = SEM(args, num_entities, num_rels, num_types)

    if use_cuda:
        model = model.to(device)
    for name, param in model.named_parameters():
        logging.debug('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

    current_learning_rate = args['lr']
    warm_up_steps = args['warm_up_steps']
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_learning_rate
    )
    tokenizer = AutoTokenizer.from_pretrained(args['plm'])
    max_valid_mrr = 0
    max_test_mrr = 0
    model.train()

    def tokenize_with_mask(sample_kg_content):
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        sep_token = tokenizer.sep_token

        curr_batch_size = sample_kg_content.shape[0]
        num_neighbors_sampled = sample_kg_content.shape[1]
        #   load text / descriptions of entities / relations from sample_kg_content
        #   heads, rels, tails.shape = args['train_batch_size'] x args['sample_kg_size']
        heads, rels, tails, = sample_kg_content[:, :, 0], sample_kg_content[:, :, 1], sample_kg_content[:, :, 2]
        heads = heads.reshape(curr_batch_size * num_neighbors_sampled)
        rels = rels.reshape(curr_batch_size * num_neighbors_sampled)
        tails = tails.reshape(curr_batch_size * num_neighbors_sampled)
        head_context = np.empty(curr_batch_size * num_neighbors_sampled, dtype=object)
        # rel_context = np.empty(curr_batch_size * num_neighbors_sampled, dtype=object)
        tail_context = np.empty(curr_batch_size * num_neighbors_sampled, dtype=object)
        #   ATTENTION: special arrangement when len(heads) == len(rels) == len(tails) == 1
        if heads.shape[0] == 1:
            head = heads[0]
            rel = rels[0]
            tail = tails[0]
            if rel < num_rels:
                kg_sequence = [cls_token + ' ' + mask_token + ' ' + sep_token + ' ' + r2text[int(rel) % num_rels] + ' ' + sep_token + ' '  + e2desc[int(tail)] + ' ' + sep_token]
            else:
                kg_sequence = [cls_token + ' ' + e2desc[int(tail)] + ' ' + sep_token + ' ' + r2text[int(rel) % num_rels] + ' ' + sep_token + ' ' + mask_token + ' ' + sep_token]
        else:
            #   if rel < num_rels, then mask head, and keep rel and tail
            head_context[rels < num_rels] = mask_token
            tail_context[rels < num_rels] = e2desc[np.array(tails)][rels < num_rels]
            rel_context = r2text[np.array(rels) % num_rels]
            #   if rel >= num_rels, then mask tail, keep rel and move tail entity to head
            head_context[rels >= num_rels] = e2desc[np.array(tails)][rels >= num_rels]
            tail_context[rels >= num_rels] = mask_token
            #   concatenate heads, rels, tails with [CLS] token and [SEP] separator
            cls_arr = np.array([cls_token + ' '], dtype=str).repeat(curr_batch_size * num_neighbors_sampled)
            sep_arr = np.array([' ' + sep_token + ' '], dtype=str).repeat(curr_batch_size * num_neighbors_sampled)
            last_sep_arr = np.array([' ' + sep_token], dtype=str).repeat(curr_batch_size * num_neighbors_sampled)
            kg_sequence = cls_arr + head_context + sep_arr + rel_context + sep_arr + tail_context + last_sep_arr
            #   tokenize sample_kg_content
            kg_sequence = kg_sequence.tolist()
        #   kg_seq_tokens = tokenizer.batch_encode_plus(kg_sequence, add_special_tokens=False, padding=True)
        kg_seq_tokens = tokenizer(kg_sequence, add_special_tokens=False, padding=True, return_tensors='pt')
        kg_mask_index = (kg_seq_tokens.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        #   kg_mask_index = (kg_seq_tokens.input_ids == tokenizer.mask_token_id).nonzero()
        #   kg_seq_tokens = tokenizer("Hello, my dog is cute", return_tensors="pt")
        return kg_seq_tokens, kg_mask_index

    def tokenize_known_type(sample_et_content):
        cls_token = tokenizer.cls_token
        mask_token = tokenizer.mask_token
        sep_token = tokenizer.sep_token
        start_str = cls_token + ' ' + mask_token + ' has type ' + sep_token + ' '
        end_str = ' ' + sep_token

        curr_batch_size = sample_et_content.shape[0]
        num_known_types_sampled = sample_et_content.shape[1]
        #   load hierarchical descriptions of known types of an entity from sample_et_content
        known_types = sample_et_content[:, :, 2]
        known_types = known_types.reshape(curr_batch_size * num_known_types_sampled)
        known_types = known_types - num_entities
        known_types_context = np.empty(curr_batch_size * num_known_types_sampled, dtype=object)
        if known_types.shape[0] == 1:
            known_type = known_types[0]
            et_sequence = [start_str + t2desc[int(known_type)] + end_str]
        else:
            start_arr = np.array([start_str], dtype=str).repeat(curr_batch_size * num_known_types_sampled)
            end_arr = np.array([end_str], dtype=str).repeat(curr_batch_size * num_known_types_sampled)
            known_types_context[known_types >= 0] = t2desc[np.array(known_types)]
            et_sequence = start_arr + known_types_context + end_arr
            et_sequence = et_sequence.tolist()
        et_seq_tokens = tokenizer(et_sequence, add_special_tokens=False, padding=True, return_tensors='pt')
        et_mask_index = (et_seq_tokens.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        return et_seq_tokens, et_mask_index

    for epoch in range(args['max_epoch']):
        log = []
        iter = 0
        for sample_et_content, sample_kg_content, gt_ent in train_dataloader:
            iter += 1
            bs = sample_kg_content.shape[0]
            kg_seq_tokens, kg_mask_index = tokenize_with_mask(sample_kg_content)
            et_seq_tokens, et_mask_index = tokenize_known_type(sample_et_content)
            type_label = train_type_label[gt_ent, :]
            if use_cuda:
                kg_seq_tokens = kg_seq_tokens.to(device)
                et_seq_tokens = et_seq_tokens.to(device)
                #   kg_mask_index = kg_mask_index.to(device)
                type_label = type_label.to(device)
            type_predict = model(kg_seq_tokens, kg_mask_index, et_seq_tokens, et_mask_index, bs)

            # if use_cuda:
            #     sample_et_content = sample_et_content.cuda()
            #     sample_kg_content = sample_kg_content.cuda()
            #     type_label = type_label.cuda()
            # type_predict = model(sample_et_content, sample_kg_content, sample_ent2pair)

            if args['loss'] == 'BCE':
                bce_loss = torch.nn.BCELoss()
                type_loss = bce_loss(type_predict, type_label)
                type_pos_loss, type_neg_loss = type_loss, type_loss
            elif args['loss'] == 'FNA':
                type_pos_loss, type_neg_loss = fna_loss(type_predict, type_label, args['beta'])
                type_loss = type_pos_loss + type_neg_loss
            elif args['loss'] == 'SFNA':
                type_pos_loss, type_neg_loss = slight_fna_loss(type_predict, type_label, args['beta'])
                type_loss = type_pos_loss + type_neg_loss
            else:
                raise ValueError('loss %s is not defined' % args['loss'])

            log.append({
                "loss": type_loss.item(),
                "pos_loss": type_pos_loss.item(),
                "neg_loss": type_neg_loss.item(),
            })
            # logging.debug('epoch %d, iter %d: loss: %f\tpos_loss: %f\tneg_loss: %f' %
            #               (epoch, iter, type_loss.item(), type_pos_loss.item(), type_neg_loss.item()))

            optimizer.zero_grad()
            type_loss.requires_grad_(True)
            type_loss.backward()
            optimizer.step()

        if epoch >= warm_up_steps:
            # current_learning_rate = current_learning_rate / 5
            current_learning_rate = current_learning_rate / 2
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 2

        avg_type_loss = sum([_['loss'] for _ in log]) / len(log)
        avg_type_pos_loss = sum([_['pos_loss'] for _ in log]) / len(log)
        avg_type_neg_loss = sum([_['neg_loss'] for _ in log]) / len(log)
        logging.debug('epoch %d: loss: %f\tpos_loss: %f\tneg_loss: %f' %
                      (epoch, avg_type_loss, avg_type_pos_loss, avg_type_neg_loss))

        if epoch != 0 and epoch % args['valid_epoch'] == 0:
        #   if epoch % args['valid_epoch'] == 0:
            model.eval()
            with torch.no_grad():
                logging.debug('-----------------------valid step-----------------------')
                predict = torch.zeros(num_entities, num_types, dtype=torch.half)
                for sample_et_content, sample_kg_content, gt_ent in valid_dataloader:
                    bs = sample_kg_content.shape[0]
                    kg_seq_tokens, kg_mask_index = tokenize_with_mask(sample_kg_content)
                    et_seq_tokens, et_mask_index = tokenize_known_type(sample_et_content)
                    if use_cuda:
                        kg_seq_tokens = kg_seq_tokens.to(device)
                        #   kg_mask_index = kg_mask_index.to(device)
                        et_seq_tokens = et_seq_tokens.to(device)
                    predict[gt_ent] = model(kg_seq_tokens, kg_mask_index, et_seq_tokens, et_mask_index, bs).cpu().half()
                    # if use_cuda:
                    #     sample_et_content = sample_et_content.cuda()
                    #     sample_kg_content = sample_kg_content.cuda()
                    # predict[gt_ent] = model(sample_et_content, sample_kg_content, sample_ent2pair).cpu().half()
                valid_mrr = evaluate(os.path.join(data_path, 'ET_valid.txt'), predict, test_type_label, e2id, t2id)

                logging.debug('-----------------------test step-----------------------')
                predict = torch.zeros(num_entities, num_types, dtype=torch.half)
                for sample_et_content, sample_kg_content, gt_ent in test_dataloader:
                    bs = sample_kg_content.shape[0]
                    kg_seq_tokens, kg_mask_index = tokenize_with_mask(sample_kg_content)
                    et_seq_tokens, et_mask_index = tokenize_known_type(sample_et_content)
                    if use_cuda:
                        kg_seq_tokens = kg_seq_tokens.to(device)
                        #   kg_mask_index = kg_mask_index.to(device)
                        et_seq_tokens = et_seq_tokens.to(device)
                    predict[gt_ent] = model(kg_seq_tokens, kg_mask_index, et_seq_tokens, et_mask_index, bs).cpu().half()
                    # if use_cuda:
                    #     sample_et_content = sample_et_content.cuda()
                    #     sample_kg_content = sample_kg_content.cuda()
                    # predict[gt_ent] = model(sample_et_content, sample_kg_content, sample_ent2pair).cpu().half()
                test_mrr = evaluate(os.path.join(data_path, 'ET_test.txt'), predict, test_type_label, e2id, t2id)

            model.train()
            if valid_mrr < max_valid_mrr:
                logging.debug('early stop')
                break
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
                max_valid_mrr = valid_mrr
            if test_mrr < max_test_mrr:
                pass
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'test_best_model.pkl'))
                max_test_mrr = test_mrr
                # save embedding
                # entity_embedding = model.entity.detach().cpu().numpy()
                # np.save(
                #     os.path.join(save_path, 'entity_embedding'),
                #     entity_embedding
                # )
                # relation_embedding = model.relation.detach().cpu().numpy()
                # np.save(
                #     os.path.join(save_path, 'relation_embedding'),
                #     relation_embedding
                # )

    logging.debug('-----------------------best test step-----------------------')
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
        model.eval()
        predict = torch.zeros(num_entities, num_types, dtype=torch.half)
        for sample_et_content, sample_kg_content, gt_ent in test_dataloader:
            bs = sample_kg_content.shape[0]
            kg_seq_tokens, kg_mask_index = tokenize_with_mask(sample_kg_content)
            et_seq_tokens, et_mask_index = tokenize_known_type(sample_et_content)
            if use_cuda:
                kg_seq_tokens = kg_seq_tokens.to(device)
                #   kg_mask_index = kg_mask_index.to(device)
                et_seq_tokens = et_seq_tokens.to(device)
            predict[gt_ent] = model(kg_seq_tokens, kg_mask_index, et_seq_tokens, et_mask_index, bs).cpu().half()
            # if use_cuda:
            #     sample_et_content = sample_et_content.cuda()
            #     sample_kg_content = sample_kg_content.cuda()
            # predict[gt_ent] = model(sample_et_content, sample_kg_content, sample_ent2pair).cpu().half()
        evaluate(os.path.join(data_path, 'ET_test.txt'), predict, test_type_label, e2id, t2id)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--save_path', type=str, default='SFNA')
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--valid_epoch', type=int, default=25)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--loss', type=str, default='SFNA')

    # params for first trm layer
    parser.add_argument('--bert_nlayer', type=int, default=3)
    parser.add_argument('--bert_nhead', type=int, default=4)
    parser.add_argument('--bert_ff_dim', type=int, default=480)
    parser.add_argument('--bert_activation', type=str, default='gelu')
    parser.add_argument('--bert_hidden_dropout', type=float, default=0.2)
    parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
    parser.add_argument('--local_pos_size', type=int, default=200)

    # params for pair trm layer
    parser.add_argument('--pair_layer', type=int, default=3)
    parser.add_argument('--pair_head', type=int, default=4)
    parser.add_argument('--pair_dropout', type=float, default=0.2)
    parser.add_argument('--pair_ff_dim', type=int, default=480)

    # params for second trm layer
    parser.add_argument('--trm_nlayer', type=int, default=3)
    parser.add_argument('--trm_nhead', type=int, default=4)
    parser.add_argument('--trm_hidden_dropout', type=float, default=0.2)
    parser.add_argument('--trm_attn_dropout', type=float, default=0.2)
    parser.add_argument('--trm_ff_dim', type=int, default=480)
    parser.add_argument('--global_pos_size', type=int, default=200)

    parser.add_argument('--pair_pooling', type=str, default='avg', choices=['max', 'avg', 'min'])
    parser.add_argument('--sample_et_size', type=int, default=3)
    parser.add_argument('--sample_kg_size', type=int, default=7)
    parser.add_argument('--sample_ent2pair_size', type=int, default=6)
    parser.add_argument('--warm_up_steps', default=50, type=int)
    parser.add_argument('--tt_ablation', type=str, default='all', choices=['all', 'triple', 'type'],
                        help='ablation choice')
    parser.add_argument('--log_name', type=str, default='log')
    parser.add_argument('--semantic', type=str, default='hybrid')

    args, _ = parser.parse_known_args()
    print(args)
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        set_logger(params)
        main(params)
    except Exception as e:
        logging.exception(e)
        raise
