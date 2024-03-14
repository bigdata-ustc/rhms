# -*- coding:utf-8 -*-

import json

def read_data_json(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return json.load(file)

def pad_sen(sen_idx_list, max_len, pad_idx):
    return sen_idx_list + [pad_idx] * (max_len - len(sen_idx_list))

def string_2_idx_sen(sen,  vocab_dict):
    if "UNK_token" in vocab_dict.keys():
        unk_idx = vocab_dict['UNK_token']
    idx_sen = [vocab_dict[word] if word in vocab_dict.keys() else unk_idx for word in sen]
    return idx_sen

def parse_sub_dependency_tree(depend2items, node):
    # depend->[(item, relation)] => (node, children, relations)
    if node in depend2items.keys():
        children = depend2items[node]
        relations = [relation for child, relation in children]
        children = [parse_sub_dependency_tree(depend2items, child) for child, relation_child2node in children]
    else:
        relations = []
        children = []
    return (node, children, relations)

def parse_dependency_tree(dependency):
    # start from 1 => start from 0
    dependency = [(relation, depend-1, item-1) for (relation, depend, item) in dependency]
    root = -1

    # [(relation, depend, item)] => depend->[(item, relation)]
    # depend->item->relation
    depend2items = dict()
    for (relation, depend, item) in dependency:
        if depend not in depend2items.keys():
            depend2items[depend] = dict()
        if item not in depend2items[depend].keys():
            depend2items[depend][item] = relation
    # depend->[(item, relation)]
    depend2items = {depend: sorted(items.items(), key=lambda kv: kv[0]) for depend, items in depend2items.items()}
    
    # (node, children, relations)
    # children: [(node, children, relations), ...]
    # only one token depend on ROOT and after ROOT
    if root in depend2items.keys() and len(depend2items[root]) == 1 and depend2items[root][0][1] == "ROOT":
        root_node, root_relation = depend2items[root][0]
        dependency_tree = parse_sub_dependency_tree(depend2items, root_node)
    else:
        dependency_tree = None
    return dependency_tree
