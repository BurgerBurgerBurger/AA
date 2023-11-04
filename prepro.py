from tqdm import tqdm
import ujson as json
import numpy as np
import pickle
import os

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}

from spacy.tokens import Doc
import spacy

nlp = spacy.load('en_core_web_sm')


def get_anaphors(sents, mentions):
    potential_mentions = []

    for sent_id, sent in enumerate(sents):
        doc_spacy = Doc(nlp.vocab, words=sent)
        for name, tool in nlp.pipeline:
            if name != 'ner':
                tool(doc_spacy)

        for token in doc_spacy:
            potential_mention = ''
            if token.dep_ == 'det' and token.text.lower() == 'the':
                potential_name = doc_spacy.text[token.idx:token.head.idx + len(token.head.text)]
                pos_start, pos_end = token.i, token.i + len(potential_name.split(' '))
                potential_mention = {
                    'pos': [pos_start, pos_end],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }
            if token.pos_ == 'PRON':
                potential_name = token.text
                pos_start = sent.index(token.text)
                potential_mention = {
                    'pos': [pos_start, pos_start + 1],
                    'type': 'MISC',
                    'sent_id': sent_id,
                    'name': potential_name
                }

            if potential_mention:
                if not any(mention in potential_mention['name'] for mention in mentions):
                    potential_mentions.append(potential_mention)

    return potential_mentions


def create_graph(entity_pos):
    anaphor_pos, entity_pos = entity_pos[-1], entity_pos[:-1]
    mention_num = len([mention for entity in entity_pos for mention in entity])
    anaphor_num = len(anaphor_pos)

    N_nodes = mention_num + anaphor_num
    nodes_adj = np.zeros((N_nodes, N_nodes), dtype=np.int32)

    edges_cnt = 1
    # add self-loop
    for i in range(N_nodes):
        nodes_adj[i, i] = edges_cnt

    edges_cnt = 2
    # add mention-anaphor edges
    for i in range(mention_num):
        for j in range(mention_num, N_nodes):
            nodes_adj[i, j] = edges_cnt
            nodes_adj[j, i] = edges_cnt

    entities = []
    i = 0
    for e in entity_pos:
        ms = []
        for _ in e:
            ms.append(i)
            i += 1
        entities.append(ms)

    edges_cnt = 3
    # add co-reference edges
    for e in entities:
        if len(e) == 1:
            continue
        for m1 in e:
            for m2 in e:
                if m1 != m2:
                    nodes_adj[m1, m2] = edges_cnt

    edges_cnt = 4
    # add inter-entity edges
    nodes_adj[nodes_adj == 0] = edges_cnt

    return nodes_adj


def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        # add * marks to the beginning and end of entities
        new_map = {}

        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end

        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos


def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int,
                        tokenizer=None):
    ''' Construct pseudo documents from predictions.'''

    pos_samples = 0
    neg_samples = 0

    sent_grps = []
    pseudo_features = []

    for pred_rel in pred_rels:
        curr_sents = pred_rel["evidence"]  # evidence sentence
        if len(curr_sents) == 0:
            continue

        # check if head/tail entity presents in evidence. if not, append sentence containing the first mention of head/tail into curr_sents
        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]])
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]])

        if len(set(head_sents) & set(curr_sents)) == 0:
            curr_sents.append(head_sents[0])
        if len(set(tail_sents) & set(curr_sents)) == 0:
            curr_sents.append(tail_sents[0])

        curr_sents = sorted(set(curr_sents))
        if curr_sents in sent_grps:  # skip if such sentence group has already been created
            continue
        sent_grps.append(curr_sents)

        # new sentence masks and input ids
        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents]
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos]
        new_input_ids = sum(new_input_ids_each, [])
        new_input_ids = tokenizer.build_inputs_with_special_tokens(new_input_ids)

        new_sent_pos = []

        prev_len = 0
        for sent in old_sent_pos:
            curr_sent_pos = (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        # iterate through all entities, keep only entities with mention in curr_sents.

        # obtain entity positions w.r.t whole document
        curr_entities = []
        ent_new2old = {}  # head/tail of a relation should be selected
        new_entity_pos = []

        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                if mention["sent_id"] in curr_sents:
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0]
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in
                           mention['pos']]
                    curr_pos.append(pos)

            if curr:
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                ent_new2old[len(ent_new2old)] = i  # update dictionary

        # check if anaphor is in pseudo document
        anaphor_in_pseudo = False
        if not entities[-1]:
            new_entity_pos.append([])
            curr_entities.append([])
        else:
            for e in curr_entities[-1]:
                for anaphor in entities[-1]:
                    if e['name'] == anaphor['name']:
                        anaphor_in_pseudo = True
            if anaphor_in_pseudo is False:
                new_entity_pos.append([])
                curr_entities.append([])

        # iterate through all entities to obtain all entity pairs
        new_hts = []
        new_labels = []
        for h in range(len(curr_entities) - 1):
            for t in range(len(curr_entities) - 1):
                if h != t:
                    new_hts.append([h, t])
                    old_h, old_t = ent_new2old[h], ent_new2old[t]
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])]
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0]
                    pos_samples += 1 - curr_label[0]

        graph = create_graph(new_entity_pos)

        pseudo_feature = {'input_ids': new_input_ids,
                          'entity_pos': new_entity_pos if new_entity_pos[-1] != [] else new_entity_pos[:-1],
                          'labels': new_labels,
                          'hts': new_hts,
                          'sent_pos': new_sent_pos,
                          'sent_labels': None,
                          'title': raw_feature['title'],
                          'entity_map': ent_new2old,
                          'graph': graph
                          }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples


def read_docred(file_in,
                tokenizer,
                transformer_type="bert",
                max_seq_length=1024,
                teacher_sig_path="",
                single_results=None):

    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []

    if file_in == "":
        return None

    with open(file_in, "r", encoding='utf-8') as fh:
        data = json.load(fh)

    if teacher_sig_path != "":  # load logits
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        attns = pickle.load(open(attns_file, 'rb'))

    if single_results != None:
        # reorder predictions as relations by title
        pred_pos_samples = 0
        pred_neg_samples = 0
        pred_rels = single_results
        title2preds = {}
        for pred_rel in pred_rels:
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel)
            else:
                title2preds[pred_rel["title"]] = [pred_rel]

    for doc_id in tqdm(range(len(data)), desc="Loading examples"):

        sample = data[doc_id]
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        # add entity markers
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}

        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])

                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        # get anaphors in the doc
        mentions = set([m['name'] for e in entities for m in e])

        potential_mention = get_anaphors(sample['sents'], mentions)

        entities.append(potential_mention)

        # entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end,))

        relations, hts, sent_labels = [], [], []

        for h, t in train_triple.keys():  # for every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)

            for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
                relation[mention["relation"]] = 1
                for i in mention["evidence"]:
                    sent_evi[i] += 1

            relations.append(relation)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        for h in range(len(entities) - 1):
            for t in range(len(entities) - 1):
                # all entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts:  # and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)

                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1

        graph = create_graph(entity_pos)

        assert len(relations) == (len(entities) - 1) * (len(entities) - 2)
        assert len(sents) < max_seq_length
        sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = [{'input_ids': input_ids,
                    'entity_pos': entity_pos if entity_pos[-1] != [] else entity_pos[:-1],
                    'labels': relations,
                    'hts': hts,
                    'sent_pos': sent_pos,
                    'sent_labels': sent_labels,
                    'title': sample['title'],
                    'graph': graph
                    }]

        if teacher_sig_path != '':  # add evidence distributions from the teacher model
            feature[0]['attns'] = attns[doc_id][:, :len(input_ids)]

        if single_results is not None:  # get pseudo documents from predictions of the single run
            offset = 1 if transformer_type in ["bert", "roberta"] else 0
            if sample["title"] in title2preds:
                feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]],
                                                                       entities, sent_map, offset, tokenizer)
                pred_pos_samples += pos_sample
                pred_neg_samples += neg_sample

        i_line += len(feature)
        features.extend(feature)

    print("# of documents {}.".format(i_line))
    if single_results is not None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))

    else:
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))

    return features
