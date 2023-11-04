import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from graph import AttentionGCNLayer


class DocREModel(nn.Module):

    def __init__(self, args, config, model, tokenizer,
                 emb_size=768, block_size=64, num_labels=-1,
                 max_sent_num=25, evi_thresh=0.2):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.use_graph = args.use_graph
        if self.use_graph:
            self.head_extractor = nn.Linear(3 * config.hidden_size, emb_size)
            self.tail_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        self.edges = ['self-loop', 'mention-anaphor', 'co-reference', 'inter-entity']

        if self.use_graph:
            self.graph_layers = nn.ModuleList(
                AttentionGCNLayer(self.edges, self.hidden_size, nhead=args.attn_heads, iters=args.gcn_layers) for _ in
                range(args.iters))

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)):  # for each batch
            entity_embs, entity_atts = [], []

            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e):  # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)  # average over all heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch

    def graph(self, sequence_output, graphs, attention, entity_pos, hts, offset):
        n, h, _, c = attention.size()

        max_node = max([graph.shape[0] for graph in graphs])
        graph_fea = torch.zeros(n, max_node, self.config.hidden_size, device=sequence_output.device)
        graph_adj = torch.zeros(n, max_node, max_node, device=sequence_output.device)

        for i, graph in enumerate(graphs):
            nodes_num = graph.shape[0]
            graph_adj[i, :nodes_num, :nodes_num] = torch.from_numpy(graph)

        for i in range(len(entity_pos)):
            mention_index = 0
            for e in entity_pos[i]:
                for start, end in e:
                    if start + offset < c:
                        # In case the entity mention is truncated due to limited max seq length.
                        graph_fea[i, mention_index, :] = sequence_output[i, start + offset]
                    else:
                        graph_fea[i, mention_index, :] = torch.zeros(self.config.hidden_size).to(sequence_output)
                    mention_index += 1

        for graph_layer in self.graph_layers:
            graph_fea, _ = graph_layer(graph_fea, graph_adj)

        h_entity, t_entity = [], []
        for i in range(len(entity_pos)):
            entity_embs = []
            mention_index = 0
            for e in entity_pos[i]:
                e_emb = graph_fea[i, mention_index:mention_index + len(e), :]
                mention_index += len(e)

                e_emb = torch.logsumexp(e_emb, dim=0) if len(e) > 1 else e_emb.squeeze(0)
                entity_embs.append(e_emb)

            entity_embs = torch.stack(entity_embs, dim=0)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_entity.append(hs)
            t_entity.append(ts)

        h_entity = torch.cat(h_entity, dim=0)
        t_entity = torch.cat(t_entity, dim=0)
        return h_entity, t_entity

    def forward_rel(self, hs, ts, rs, h, t):
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs, h], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs, t], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        return logits

    def forward_rel_no_graph(self, hs, ts, rs):
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        return logits

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):  # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]  # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent],
                                   dim=-1)  # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,  # relation labels
                entity_pos=None,
                hts=None,  # entity pairs
                sent_pos=None,
                sent_labels=None,  # evidence labels (0/1)
                teacher_attns=None,  # evidence distribution from teacher model
                graph=None,
                tag="train"
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, offset)

        if self.use_graph:
            h, t = self.graph(sequence_output, graph, attention, entity_pos, hts, offset)
            logits = self.forward_rel(hs, ts, rs, h, t)
        else:
            logits = self.forward_rel_no_graph(hs, ts, rs)

        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if sent_labels is not None:  # human-annotated evidence available

            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]:  # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]

        if tag == "infer":  # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else:  # training
            # relation extraction loss
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}

            if sent_labels is not None:  # supervised training with human evidence

                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

            elif teacher_attns is not None:  # self training with teacher attention

                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

        return output
