# -*- coding: utf-8 -*-

from copy import deepcopy
import json
import math
import numpy as np
import os
import pickle

# load/store json and pkl data

def load(path):
    with open(path, "rt", encoding="utf-8") as file:
        js = json.load(file)
    return js

def load_pickle(path):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj

def save_pickle(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    return

# parse dependency list as a tree
# (pos_tag, rel_to_parent, children_nodes)

def parse_sub_dependency_tree(node_list, node_idx, relation):
    label, children = node_list[node_idx]
    children_nodes = []
    children = sorted(children, key=lambda idx_rel: idx_rel[0])
    for child_idx, child_relation in children:
        children_nodes.append(parse_sub_dependency_tree(node_list, child_idx, child_relation))
    tree_node = (label, relation, children_nodes)
    return tree_node

def parse_dependency_tree(dependency, pos_list):
    # word_idx -> (pos_tag, [(child_idx, rel_to_child)])
    # use pos_tag as node label instead of real word
    node_list = [(pos, []) for pos in pos_list]
    root = None
    for relation, parent, child in dependency:
        # word_idx start from 1 => start from 0 to match node_list
        parent -= 1
        child -= 1
        if relation == "ROOT":
            root = child
        else:
            node_list[parent][1].append((child, relation))
    # use root node to access the tree
    tree = parse_sub_dependency_tree(node_list, root, "root")
    return tree

# dependency tree -> length-specific path_list from ancestor to offspring
# (n1, r1, n2, r2, n3)
# by moving on the tree: (n1, r1, n2, r2, n3) -> (n2, r2, n3, r3, n4) -> (n3, r3, n4, r4, n5)

# initial path generator: yield one initial path in each call
def get_init_sub_path_step(node, path, step, step_size):
    label, relation, children = node
    # current node
    if step == 1:
        if relation == "root":
            path.append(label)
        else:
            path = None
    else:
        path.append(relation)
        path.append(label)

    # expand current path by one step recursively
    if path is not None and step < step_size and len(children) > 0:
        for child in children:
            # recursively call due to tree branches
            rec_generator = get_init_sub_path_step(child, path, step + 1, step_size)
            # evoke generator
            for g in rec_generator:
                yield g
    # end expand, yield initial path
    # return children as direction to move initial path
    else:
        if path is not None and len(path) > 0:
            yield deepcopy(path), children
    
    # recover path for backtrace of recursion
    if step > 1:
        path.pop()
        path.pop()
    return

# generator: initial call of recursion
def get_init_sub_path(tree, step_size):
    return get_init_sub_path_step(tree, [], 1, step_size)

# moving path generator: move path forward by one-step in each call (or backtrace to other branches)
def get_mov_sub_path_step(node, path):
    # move node window 1 step: pop 1 node from start, append 1 node at end
    # yield one-step moved path
    label, relation, children = node
    start_label = path.pop(0)
    start_relation = path.pop(0)
    path.append(relation)
    path.append(label)
    yield tuple(path)

    # move ahead
    if len(children) > 0:
        for child in children:
            # recursively call due to tree branches
            rec_generator = get_mov_sub_path_step(child, path)
            # evoke generator
            for g in rec_generator:
                yield g

    # recover path for backtrace of recursion
    path.pop()
    path.pop()
    path.insert(0, start_relation)
    path.insert(0, start_label)
    return

# tree -> path_list for one question
def get_tree_sub_path(tree, step_size):
    sub_path_list = []
    for path, children in get_init_sub_path(tree, step_size):
        if len(path) < step_size:
            if len(path) >= 2:
                sub_path_list.append(tuple(path))
        else:
            sub_path_list.append(tuple(path))
            if len(children) > 0:
                for child in children:
                    for p in get_mov_sub_path_step(child, path):
                        sub_path_list.append(p)
    return sub_path_list

# parse dependency and generate path_list for whole dataset
def dataset_sub_path(path, save_path, step_size=3):
    dataset = load(path)
    q2subpath = dict()
    for item in dataset:
        qid = item["id"]
        pos = item["pos_tag"]
        dependency = item["dependency"]
        sub_path_list = []
        for p, d in zip(pos, dependency):
            tree = parse_dependency_tree(json.loads(d), p.strip().split(' '))
            sub_paths = get_tree_sub_path(tree, step_size)
            sub_path_list.extend(sub_paths)
        q2subpath[qid] = sub_path_list
    save_pickle(q2subpath, save_path)
    return

# compute tf-idf vector for path_list
# generate one tf-idf vector for one path_list, each path is an element of the vector

# compute idf value for each path globally on the whole dataset
def get_idf(corpus):
    w2df = dict()
    for s in corpus:
        for w in set(s):
            w2df[w] = w2df.get(w, 0) + 1
    w2idf = {w:1+math.log(len(corpus)/df) for w, df in w2df.items()}
    return w2idf

# compute tf-idf vector for each path_list locally with global idf-value
# tf-idf vector: path->value dict due to very large vocab of path
def get_tf_idf(s, w2idf):
    w2tf = dict()
    for w in s:
        w2tf[w] = w2tf.get(w, 0) + 1
    # normalized tf
    w2tdf = {w:tf/len(s)*w2idf[w] for w, tf in w2tf.items()}
    return w2tdf

# compute tf-idf vector for whole dataset based on path_list
def tf_idf(load_path, save_path):
    q2subpath = load_pickle(load_path)
    subpath2idf = get_idf(q2subpath.values())
    q2tfidf = dict()
    for qid, subpath in q2subpath.items():
        tfidf = get_tf_idf(subpath, subpath2idf)
        q2tfidf[qid] = tfidf
    save_pickle(q2tfidf, save_path)
    return

# parse_dependency, generate path_list, and compute tf-idf for whole dataset
def dataset_tf_idf(path, sub_path_path, tf_idf_path, step_size=3):
    dataset_sub_path(path, sub_path_path, step_size)
    tf_idf(sub_path_path, tf_idf_path)
    return

# compute similarity based on path_list tf-idf vector

def sub_path_similarity(tfidf1, tfidf2):
    vec1 = []
    vec2 = []
    for subpath in (tfidf1.keys() | tfidf2.keys()):
        vec1.append(tfidf1.get(subpath, 0))
        vec2.append(tfidf2.get(subpath, 0))
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # cosine
    similarity = (vec1 * vec2).sum() / (np.linalg.norm(vec1, 2) * np.linalg.norm(vec2, 2))
    return similarity

# compute train-train similarity
def dataset_similarity(tf_idf_path, similarity_path):
    q2tfidf = list(load_pickle(tf_idf_path).items())
    print("calculating similarity ...")
    similarities = []
    for i, (qid1, tfidf1) in enumerate(q2tfidf):
        if i % 500 == 0:
            print(f"\t{i} ...")
        for qid2, tfidf2 in q2tfidf[i+1:]:
            similarity = sub_path_similarity(tfidf1, tfidf2)
            similarities.append((qid1, qid2, similarity))
    print("sorting similarity ...")
    similarities.sort(key=lambda x:x[2], reverse=True)
    print("saving similarity ...")
    with open(similarity_path, "wt", encoding="utf-8") as file:
        for qid1, qid2, similarity in similarities:
            file.write(f"{qid1}\t{qid2}\t{similarity}\n")
    return

# compute train-test similarity
def dataset_similarity_train_test(train_tf_idf_path, test_tf_idf_path, similarity_path):
    train_tfidf = list(load_pickle(train_tf_idf_path).items())
    test_tfidf = list(load_pickle(test_tf_idf_path).items())
    print("calculating similarity ...")
    similarities = []
    for i, (qid1, tfidf1) in enumerate(test_tfidf):
        if i % 50 == 0:
            print(f"\t{i} ...")
        for qid2, tfidf2 in train_tfidf:
            similarity = sub_path_similarity(tfidf1, tfidf2)
            similarities.append((qid1, qid2, similarity))
    print("sorting similarity ...")
    similarities.sort(key=lambda x:x[2], reverse=True)
    print("saving similarity ...")
    with open(similarity_path, "wt", encoding="utf-8") as file:
        for qid1, qid2, similarity in similarities:
            file.write(f"{qid1}\t{qid2}\t{similarity}\n")
    return

# construct graph based on similarity
# only question pair with similarity larger than threshold

def construct_graph(similarity_path, sim_threshold):
    graph = {}
    with open(similarity_path, "rt", encoding="utf-8") as file:
        for line in file:
            qid1, qid2, sim = line.strip('\n').split('\t')
            if qid1 not in graph.keys():
                graph[qid1] = []
            if qid2 not in graph.keys():
                graph[qid2] = []
            sim = float(sim)
            if sim >= sim_threshold:
                graph[qid1].append(qid2)
                graph[qid2].append(qid1)
    for edges in graph.values():
        edges.sort()
    return graph

def save_graph(graph, path):
    graph = sorted(graph.items(), key=lambda kv: kv[0])
    with open(path, "wt", encoding="utf-8") as file:
        for node, edges in graph:
            edges = [node] + edges
            file.write(' '.join(edges) + '\n')
    return

def load_graph(path):
    graph = dict()
    with open(path, "rt", encoding="utf-8") as file:
        for line in file:
            edges = line.strip('\n').split(' ')
            node = edges[0]
            edges = edges[1:]
            graph[node] = edges
    return graph

if __name__ == "__main__":
    data_path = "data"
    temp_path = os.path.join(data_path, "temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    
    # generate tf-idf vector
    for dataset in ["train", "test", "valid"]:
        path = os.path.join(data_path, f"{dataset}23k_processed.json")
        sub_path_path = os.path.join(temp_path, f"{dataset}_sub_path.pickle")
        tf_idf_path = os.path.join(temp_path, f"{dataset}_tf_idf.pickle")
        dataset_tf_idf(path, sub_path_path, tf_idf_path, step_size=3)
    
    # compute similarity
    # train-train
    sim_path = os.path.join(temp_path, "train_sim.txt")
    tf_idf_path = os.path.join(temp_path, "train_tf_idf.pickle")
    dataset_similarity(tf_idf_path, sim_path)
    # train-test
    for dataset in ["test", "valid"]:
        sim_path = os.path.join(temp_path, f"{dataset}_sim.txt")
        test_tf_idf_path = os.path.join(temp_path, f"{dataset}_tf_idf.pickle")
        train_tf_idf_path = os.path.join(temp_path, "train_tf_idf.pickle")
        dataset_similarity_train_test(train_tf_idf_path, test_tf_idf_path, sim_path)

    # construct graph
    threshold = 0.85
    for dataset in ["train", "test", "valid"]:
        sim_path = os.path.join(temp_path, f"{dataset}_sim.txt")
        graph_path = os.path.join(data_path, f"{dataset}_graph_{threshold:.2f}.txt")
        save_graph(construct_graph(sim_path, threshold), graph_path)
