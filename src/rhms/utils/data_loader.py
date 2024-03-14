# -*- coding:utf-8 -*-

from copy import deepcopy
import json
import logging
import os
import random
import torch

from .data_tools import read_data_json, string_2_idx_sen, pad_sen, parse_dependency_tree

def convert_dependency_tree(str_dependencies):
    dependency = [parse_dependency_tree(json.loads(str_dependency)) for str_dependency in str_dependencies]
    return dependency

def load_similarity_graph(path):
    graph = dict()
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            edges = line.strip('\n').split(' ')
            node = edges[0]
            edges = edges[1:]
            graph[node] = edges
    return graph

class DataLoader:
    def __init__(self, mode, hop, threshold, trim_min_count=5, embed_dim=128):
        self.hop = hop

        train_list = read_data_json("data/train23k_processed.json")
        valid_list = read_data_json("data/valid23k_processed.json")
        test_list = read_data_json("data/test23k_processed.json")
        if self.hop > 0:
            self.train_graph = load_similarity_graph(f"data/train_graph_{threshold:.2f}.txt")
            self.test_graph = load_similarity_graph(f"data/test_graph_{threshold:.2f}.txt")
        self.wv_path = "data/word2vec.pt"

        self.op_set = set("+-*/^")
        self.train_list = train_list
        self.test_list = test_list
        self.train_id_list = set(data["id"] for data in self.train_list)
        logging.info(f"train size: {len(self.train_list)}, test size: {len(self.test_list)}")

        # build vocab
        build_wv = not os.path.exists(self.wv_path)
        embed_vectors, self.vocab_list, self.class_list, self.span_size = self.preprocess_vocab(embed_dim, trim_min_count, build_wv=build_wv)
        if build_wv:
            self.embed_vectors = embed_vectors
            torch.save(self.embed_vectors, self.wv_path)
        else:
            self.embed_vectors = torch.load(self.wv_path)
        
        self.vocab_dict = {token: idx for idx, token in enumerate(self.vocab_list)}
        self.vocab_len = len(self.vocab_list)
        self.class_dict = {token: idx for idx, token in enumerate(self.class_list)}

        self.test_list = self.preprocess_dataset(self.test_list)
        if mode == "train" or self.hop > 0:
            self.train_list = self.preprocess_dataset(self.train_list)
            if self.hop > 0:
                self.train_dict = {data["index"]:data for data in self.train_list}
        return

    def preprocess_vocab(self, embed_dim, trim_min_count, build_wv=False, wv_epoch=30):
        # word count
        sentences = []
        word_count = {}
        equ_op_tokens = set()
        equ_const_tokens = set()
        num_tokens = set()

        span_size = max(len(data["spans"]) for data in self.train_list)

        for data in self.train_list:
            sentence = [word for span in data["spans"] for word in span.strip().split(' ')]
            for word in sentence:
                if "temp_" in word and word not in num_tokens:
                    num_tokens.add(word)
            
            sentence = [word if "temp_" not in word else 'NUM_token' for word in sentence]
            sentences.append(sentence)
            for word in sentence:
                word_count[word] = word_count.get(word, 0) + 1

            equation = data["target_norm_pre_template"]
            if type(equation) is str:
                equation = equation.split(' ')
            equation = equation[2:]
            duplicate_num = json.loads(data["duplicate_num"])
            duplicate_num = [v for vs in duplicate_num.values() for v in vs]
            equation += duplicate_num
            for token in equation:
                if "temp_" not in token:
                    if token in self.op_set:
                        equ_tokens = equ_op_tokens
                    else:
                        equ_tokens = equ_const_tokens
                    if token not in equ_tokens:
                        equ_tokens.add(token)
        
        for data in self.test_list:
            sentence = [word for span in data["spans"] for word in span.strip().split(' ')]
            for word in sentence:
                if "temp_" in word and word not in num_tokens:
                    num_tokens.add(word)
            equation = data["target_norm_pre_template"]
            if type(equation) is str:
                equation = equation.split(' ')
            equation = equation[2:]
            duplicate_num = json.loads(data["duplicate_num"])
            duplicate_num = [v for vs in duplicate_num.values() for v in vs]
            equation += duplicate_num
            for token in equation:
                if "temp_" not in token:
                    if token in self.op_set:
                        equ_tokens = equ_op_tokens
                    else:
                        equ_tokens = equ_const_tokens
                    if token not in equ_tokens:
                        equ_tokens.add(token)
        
        # word trim
        if build_wv:
            from gensim.models import word2vec
            import numpy as np
            model = word2vec.Word2Vec(sentences, size=embed_dim, min_count=1, epochs=wv_epoch, sg=1)

        # unused END_token
        vocab_list = ['PAD_token', 'UNK_token', 'END_token']
        class_list = ['PAD_token']
        if build_wv:
            embed_vectors = []
            embed_vectors.append(np.zeros(embed_dim))           # PAD_token
            embed_vectors.append(np.random.rand(embed_dim))     # UNK_token
            embed_vectors.append(np.random.rand(embed_dim))     # END_token

        for word, cnt in sorted(word_count.items(), key=lambda wc: wc[1], reverse=True):
            if cnt >= trim_min_count and word not in vocab_list:
                vocab_list.append(word)
                if build_wv:
                    embed_vectors.append(np.array(model.wv[word]))
        class_list = class_list + sorted(equ_op_tokens) + sorted(equ_const_tokens) + sorted(num_tokens)
        
        # coverage statistics
        vocab_set = set(vocab_list)
        n_all_word = 0
        n_cov_word = 0
        for data in self.train_list:
            for span in data["spans"]:
                for word in span.split(' '):
                    n_all_word += 1
                    if word in vocab_set:
                        n_cov_word += 1
        
        logging.info(f"saving vocab({trim_min_count}): {len(vocab_list)-3} / {len(word_count)} = {(len(vocab_list)-3)/len(word_count):.4f}, coverage {n_cov_word/n_all_word:.4f}")
        
        # class token embedding
        for token in class_list:
            if token not in vocab_list:
                vocab_list.append(token)
                if build_wv:
                    embed_vectors.append(np.random.rand(embed_dim))
        
        if build_wv:
            embed_vectors = torch.tensor(np.array(embed_vectors))
        else:
            embed_vectors = None

        logging.info(f"vocab_len: {len(vocab_list)}, classes_len: {len(class_list)}")
        logging.info(f"decode_classes: {' '.join(class_list)}")
        logging.info(f"max_span_len: {span_size}")
        return embed_vectors, vocab_list, class_list, span_size

    def preprocess_dataset(self, dataset):
        data_dataset = []
        for data in dataset:
            # index
            index = str(data['id'])
            # num_list
            num_list = data['num_list']
            # answer
            solution = data['answer']
            # text
            encode_text = ' '.join(data["spans"])

            # span
            raw_spans = [span.strip().split(' ') for span in data["spans"]]
            encode_spans = [[word if "temp_" not in word else 'NUM_token' for word in span] for span in raw_spans]
            words = [word for span in encode_spans for word in span]
            encode_spans_idx = [string_2_idx_sen(span, self.vocab_dict) for span in encode_spans]
            encode_spans_len = [len(span) for span in encode_spans]
            span_len = len(encode_spans)
            
            # target
            decode_sen = data['target_norm_pre_template']
            if type(decode_sen) is str:
                decode_sen = decode_sen.split(' ')
            decode_sen = decode_sen[2:]
            decode_sen_idx = string_2_idx_sen(decode_sen, self.class_dict)
            decode_len = len(decode_sen_idx)
            # duplicate num
            duplicate_num = json.loads(data["duplicate_num"])
            duplicate_num = {self.class_dict[k]:[self.class_dict[v] for v in vs] for k, vs in duplicate_num.items()}
            
            # num_pos, span_num_pos, word_num_pos
            span_num_pos = [-1] * len(self.class_list)
            word_num_poses = [[-1] * len(self.class_list) for _ in range(len(raw_spans))]
            for i, span in enumerate(raw_spans):
                for j, word in enumerate(span):
                    if "temp_" in word and word in self.class_dict.keys():
                        class_index = self.class_dict[word]
                        span_num_pos[class_index] = i
                        word_num_poses[i][class_index] = j
        
            data_elem = dict()
            data_elem['index'] = index
            data_elem['text'] = encode_text
            data_elem['words'] = words
            data_elem['num_list'] = num_list
            data_elem['solution'] = solution

            data_elem['span_encode_idx'] = encode_spans_idx
            data_elem['span_encode_len'] = encode_spans_len
            data_elem['span_len'] = span_len
            
            data_elem['span_num_pos'] = span_num_pos
            data_elem['word_num_poses'] = word_num_poses
            
            # tree
            data_elem['tree'] = convert_dependency_tree(data["dependency"])
            
            data_elem['decode_idx'] = decode_sen_idx
            data_elem['decode_len'] = decode_len
            data_elem['duplicate_num'] = duplicate_num
            
            data_dataset.append(data_elem)
        return data_dataset
          
    def process_data_batch(self, data_batch, template_flag):
        batch_idxs = []
        batch_text = []
        batch_words = []
        batch_num_list = []
        batch_solution = []

        batch_span_encode_idx = []
        batch_span_encode_len = []
        batch_span_len = []

        batch_decode_idx = []
        batch_decode_len = []
        batch_duplicate_num = []

        batch_span_num_pos = []
        batch_word_num_poses = []
        batch_tree = []

        for data in data_batch:
            # id
            batch_idxs.append(data["index"])
            # text
            batch_text.append(data["text"])
            batch_words.append(data["words"])
            # num_list
            batch_num_list.append(data['num_list'])
            # answer
            batch_solution.append(data['solution'])

            # spans
            # batch * span
            batch_span_encode_idx.append(data["span_encode_idx"])
            batch_span_encode_len.append(data["span_encode_len"])
            batch_span_len.append(data["span_len"])
            
            # target
            if template_flag:
                batch_decode_idx.append(data["decode_idx"])
                batch_decode_len.append(data["decode_len"])
                batch_duplicate_num.append(deepcopy(data["duplicate_num"]))

            # num_pos, span_num_pos, word_num_poses
            batch_span_num_pos.append(data["span_num_pos"])
            # batch * span
            batch_word_num_poses.append(data["word_num_poses"])
            
            # dependency
            # batch * span
            batch_tree.append(data["tree"])

        # pad
        # max_len
        if len(data_batch) > 0:
            max_span_len = max(batch_span_len)
            max_span_encode_len = [max(elem_span_len[i] for elem_span_len in batch_span_encode_len if i < len(elem_span_len)) for i in range(max_span_len)]
            if template_flag:
                max_decode_len = max(batch_decode_len)
        else:
            max_span_len = 0
            max_span_encode_len = []
            if template_flag:
                max_decode_len = 0
        
        # span
        # span * batch
        batch_span_encode_idx_pad = [[] for _ in range(max_span_len)]
        batch_span_encode_len_pad = [[] for _ in range(max_span_len)]       
        batch_word_num_poses_pad = [[] for _ in range(max_span_len)]
        batch_tree_pad = [[] for _ in range(max_span_len)]
        # decode
        if template_flag:
            batch_decode_idx_pad = []

        pad_spans = [[self.vocab_dict['PAD_token']] * encode_len for encode_len in max_span_encode_len]
        pad_num_pos = [-1] * len(self.class_list)
        for data_index in range(len(data_batch)):
            # spans
            span_len = batch_span_len[data_index]
            encode_spans = batch_span_encode_idx[data_index]
            encode_lens = batch_span_encode_len[data_index]
            for span_index in range(max_span_len):
                max_encode_len = max_span_encode_len[span_index]
                if span_index < span_len:
                    encode_span = encode_spans[span_index]
                    encode_span = pad_sen(encode_span, max_encode_len, self.vocab_dict['PAD_token'])
                    encode_len = encode_lens[span_index]
                else:
                    encode_span = pad_spans[span_index]
                    encode_len = 0
                batch_span_encode_idx_pad[span_index].append(encode_span)
                batch_span_encode_len_pad[span_index].append(encode_len)
            
            if template_flag:
                # target
                decode_sen_idx = batch_decode_idx[data_index]
                decode_sen_idx_pad = pad_sen(decode_sen_idx, max_decode_len, self.class_dict['PAD_token'])
                batch_decode_idx_pad.append(decode_sen_idx_pad)
            
            # word_num_poses
            word_num_poses = batch_word_num_poses[data_index]
            for span_index in range(max_span_len):
                if span_index < span_len:
                    word_num_pos = word_num_poses[span_index]
                else:
                    word_num_pos = pad_num_pos
                batch_word_num_poses_pad[span_index].append(word_num_pos)
            
            # dependency
            trees = batch_tree[data_index]
            for span_index in range(max_span_len):
                # tree
                if span_index < span_len:
                    tree = trees[span_index]
                else:
                    tree = None
                batch_tree_pad[span_index].append(tree)

        batch_data_dict = dict()
        batch_data_dict['batch_index'] = batch_idxs
        batch_data_dict['batch_text'] = batch_text
        batch_data_dict['batch_words'] = batch_words
        batch_data_dict['batch_num_list'] = batch_num_list
        batch_data_dict['batch_solution'] = batch_solution

        batch_data_dict["batch_span_encode_idx"] = batch_span_encode_idx_pad
        batch_data_dict["batch_span_encode_len"] = batch_span_encode_len_pad
        batch_data_dict["batch_span_len"] = batch_span_len

        batch_data_dict["batch_span_num_pos"] = batch_span_num_pos
        batch_data_dict["batch_word_num_poses"] = batch_word_num_poses_pad
        
        batch_data_dict["batch_tree"] = batch_tree_pad

        if template_flag:
            batch_data_dict['batch_decode_idx'] = batch_decode_idx_pad
            batch_data_dict['batch_duplicate_num'] = batch_duplicate_num
        return batch_data_dict
    
    def collect_batch(self, batch_data_list, dataset_flag):
        graph_map = []
        batch_neighbor_list = []
        if self.hop > 0:
            batch_idx_list = [data["index"] for data in batch_data_list]
            neighbor_idx_list = []
            # neighbor data append after batch data in model
            neighbor_start_pos = len(batch_idx_list)
            for hop_idx in range(self.hop):
                if hop_idx == 0:
                    from_idx_list = batch_idx_list
                else:
                    from_idx_list = new_neighbor_idx_list
                new_neighbor_idx_list = []
                for from_idx in from_idx_list:
                    neighbor_map = []
                    if hop_idx == 0 and dataset_flag == "test":
                        neighbors = self.test_graph[from_idx]
                    else:
                        neighbors = self.train_graph[from_idx]
                    neighbors = [neighbor_idx for neighbor_idx in neighbors if neighbor_idx in self.train_id_list]
                    for neighbor_idx in neighbors:
                        if neighbor_idx in batch_idx_list:
                            neighbor_pos = batch_idx_list.index(neighbor_idx)
                        elif neighbor_idx in neighbor_idx_list:
                            neighbor_pos = neighbor_start_pos + neighbor_idx_list.index(neighbor_idx)
                        else:
                            batch_neighbor_list.append(self.train_dict[neighbor_idx])
                            neighbor_idx_list.append(neighbor_idx)
                            new_neighbor_idx_list.append(neighbor_idx)
                            neighbor_pos = neighbor_start_pos + len(neighbor_idx_list) - 1
                        neighbor_map.append(neighbor_pos)
                    graph_map.append(neighbor_map)
            # pad
            max_neighbor_len = max(len(neighbor_map) for neighbor_map in graph_map)
            graph_map = [neighbor_map+[-1]*(max_neighbor_len-len(neighbor_map)) for neighbor_map in graph_map]
        return batch_neighbor_list, graph_map

    def get_batch(self, data_list, batch_size, template_flag=False, shuffle=False):
        if data_list is self.train_list:
            dataset_flag = "train"
        elif data_list is self.test_list:
            dataset_flag = "test"
        else:
            dataset_flag = None
        batch_num = int(len(data_list)/batch_size)
        if len(data_list) % batch_size != 0:
            batch_num += 1
        if shuffle:
            # avoid shuffle original dataset
            data_list = [item for item in data_list]
            random.shuffle(data_list)
        for idx in range(batch_num):
            batch_start = idx*batch_size
            batch_end = min((idx+1)*batch_size, len(data_list))
            batch_data_list = data_list[batch_start: batch_end]
            batch_neighbor_list, batch_graph_map = self.collect_batch(batch_data_list, dataset_flag)
            batch_data_dict = self.process_data_batch(batch_data_list, template_flag)
            if self.hop > 0:
                batch_neighbor_dict = self.process_data_batch(batch_neighbor_list, False)
            else:
                batch_neighbor_dict = None
            yield batch_data_dict, batch_neighbor_dict, batch_graph_map
        return
