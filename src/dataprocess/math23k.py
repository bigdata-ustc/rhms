# -*- encoding:utf-8 -*-

import json
import os
import re
from stanfordcorenlp import StanfordCoreNLP

from equ_tools import infix_to_postfix, postfix_to_prefix, post_solver, number_map, eval_num_list

def read_dataset(dataset_path):
    with open(dataset_path, "rt", encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset

def save_dataset(dataset, dataset_path):
    with open(dataset_path, "wt", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
    return

def transfer_num(data):
    pattern = re.compile(r"\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    n_data = list()
    for d in data:
        nums = []
        input_seq = []
        seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        n_num = 0
        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append(f"temp_{chr(n_num + ord('a'))}")
                n_num += 1
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)

        nums_fraction = []
        for num in nums:
            if re.search(r"\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        # seg the equation and tag the num
        def seg_and_tag(st):
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if n in nums:
                        res.append(f"temp_{chr(nums.index(n) + ord('a'))}")
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search(r"\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if st_num in nums:
                    res.append(f"temp_{chr(nums.index(st_num) + ord('a'))}")
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        postfix = infix_to_postfix(out_seq)
        f_nums = eval_num_list(nums)
        n_d = dict()
        n_d['id'] = d['id']
        n_d['equation'] = d['equation']
        n_d['text'] = ' '.join(input_seq)
        n_d['target_norm_post_template'] = ' '.join(['x', '='] + postfix)
        n_d['s_num_list'] = nums
        n_d['num_list'] = f_nums
        n_d["original_text"] = d["original_text"]
        n_data.append(n_d)
        if post_solver(number_map(postfix, f_nums)) is None:
            print(d['id'])
    return n_data

def duplicate_num(data):
    # same number appear several times in question: 2 children have 2 apples
    for d in data:
        const_list = ['1', '2']
        s_num_list = d["s_num_list"]
        first2all = dict()
        any2first = dict()
        for i, item1 in enumerate(s_num_list):
            if i not in any2first.keys():
                any2first[i] = i
                i_token = f"temp_{chr(ord('a') + i)}"
                first2all[i_token] = [i_token]
                for j, item2 in enumerate(s_num_list[i+1:]):
                    j = i + 1 + j
                    if item2 == item1:
                        j_token = f"temp_{chr(ord('a') + j)}"
                        if j not in any2first.keys():
                            any2first[j] = i
                            if j_token not in first2all[i_token]:
                                first2all[i_token].append(j_token)
                if item1 in const_list:
                    first2all[i_token].append(item1)
        expr = d["target_norm_post_template"].split(' ')[2:]
        first2all = {k:vs for k, vs in first2all.items() if len(vs) > 1 and k in expr}
        d["duplicate_num"] = json.dumps(first2all)
    return data

def num_process(in_path, out_path):
    data = read_dataset(in_path)
    data = transfer_num(data)
    data = duplicate_num(data)
    save_dataset(data, out_path)
    return

def start_end_span_process(spans):
    # correctly split "“”" "（）" et al.
    start_symbols = "“‘（《(["
    end_symbols = "”’）》)]"
    for i in range(len(spans)):
        if spans[i][-1] in start_symbols:
            spans[i+1] = [spans[i][-1]] + spans[i+1]
            spans[i] = spans[i][:-1]
        if spans[i][0] in end_symbols:
            spans[i-1] += [spans[i][0]]
            spans[i] = spans[i][1:]
        if len(spans[i]) == 1:
            spans[i-1] += spans[i]
            spans[i] = None
        elif len(spans[i]) == 0:
            spans[i] = None
    new_spans = [span for span in spans if span is not None and span != '']
    return new_spans

def is_number(word):
    return word[0].isdigit() or (word[0] == '(' and word[1].isdigit())

def colon_span_process(spans):
    # correctly split "："
    new_spans = []
    for span in spans:
        split_pos = []
        if '：' in span:
            for j in range(len(span)):
                # except start, end, number ratio
                if span[j] == '：' and ((j == 0 or (not is_number(span[j-1]))) or (j == len(span) - 1 or (not is_number(span[j+1])))):
                    if j > 1 and j < len(span) - 3:
                        split_pos.append(j)
        if len(split_pos) == 0:
            new_spans.append(span)
        else:
            start_pos = 0
            for end_pos in split_pos:
                new_spans.append(span[start_pos:end_pos+1])
                start_pos = end_pos + 1
            new_spans.append(span[start_pos:])
    return new_spans

def split_text(text):
    seps = "，。．；？！!"
    sep_pattern = re.compile(f"([{seps}])", re.S)
    spans = re.split(sep_pattern, text)
    spans = [span.strip() for span in spans if span.strip() != '']
    spans_post = []
    for i, span in enumerate(spans):
        if span in seps:
            if i > 0 and spans[i - 1] not in seps:
                spans_post[-1] += ' ' + span
        else:
            spans_post.append(span)
    spans = spans_post
    seg_spans = [span.split(' ') for span in spans]
    seg_spans = start_end_span_process(seg_spans)
    seg_spans = colon_span_process(seg_spans)
    spans = [' '.join(span) for span in seg_spans]
    return spans

def text_temp_2_num(spans, num_list):
    # temp_a -> 2 before tokenization for correct pos tagging & dependency parsing
    # numbers like 0.12345678 may be mistakenly broken in tokenization
    # => use 2/0.5/1.5 to replace
    new_spans = []
    for span in spans:
        new_span = []
        for w in span.split(' '):
            if "temp_" in w:
                num = num_list[ord(w[-1]) - ord('a')]
                if '.' in str(num):
                    if num < 1:
                        repl_token = "0.5"
                    else:
                        repl_token = "1.5"
                else:
                    repl_token = "2"
                new_span.append(repl_token)
            else:
                new_span.append(w)
        new_span = ' '.join(new_span)
        new_spans.append(new_span)
    return new_spans

def text_num_2_temp(spans):
    # 2 -> temp_a after tokenization
    repl_tokens = ["2", "0.5", "1.5"]
    new_spans = []
    num_index = 0
    for span in spans:
        new_span = []
        for w in span.split(' '):
            if w in repl_tokens:
                new_span.append(f"temp_{chr(ord('a') + num_index)}")
                num_index += 1
            else:
                new_span.append(w)
        new_span = ' '.join(new_span)
        new_spans.append(new_span)
    return new_spans

def nlp_process(nlp_path, in_path, out_path):
    dataset = read_dataset(in_path)
    nlp = StanfordCoreNLP(nlp_path, lang='zh')
    for item in dataset:
        spans = split_text(item["text"])
        nlp_spans = []
        nlp_pos = []
        nlp_dep = []
        num_spans = text_temp_2_num(spans, item["num_list"])
        for span in num_spans:
            tokens, pos_tag = list(zip(*nlp.pos_tag(span)))
            dep = nlp.dependency_parse(span)
            span = ' '.join(tokens)
            pos_tag = ' '.join(pos_tag)
            dep = json.dumps(dep)
            nlp_spans.append(span)
            nlp_pos.append(pos_tag)
            nlp_dep.append(dep)
        nlp_spans = text_num_2_temp(nlp_spans)
        item["spans"] = nlp_spans
        item["pos_tag"] = nlp_pos
        item["dependency"] = nlp_dep
    nlp.close()
    save_dataset(dataset, out_path)
    return

def equ_process(in_path, out_path):
    raw_dataset = read_dataset(in_path)
    dataset = list()
    for raw_item in raw_dataset:
        post = raw_item["target_norm_post_template"].split(' ')[2:]
        num_list = raw_item["num_list"]
        pre = postfix_to_prefix(post)
        answer = post_solver(number_map(post, num_list))[1]

        out_item = dict()
        out_item["id"] = raw_item["id"]
        out_item["original_text"] = raw_item["original_text"]
        out_item["equation"] = raw_item["equation"]
        out_item["text"] = raw_item["text"]
        out_item["answer"] = answer
        out_item["s_num_list"] = raw_item["s_num_list"]
        out_item["num_list"] = num_list
        out_item["target_norm_pre_template"] = ' '.join(["x", "="] + pre)
        out_item["spans"] = raw_item["spans"]
        out_item["dependency"] = raw_item["dependency"]
        out_item["pos_tag"] = raw_item["pos_tag"]
        out_item["duplicate_num"] = raw_item["duplicate_num"]
        dataset.append(out_item)
    save_dataset(dataset, out_path)
    return

if __name__ == "__main__":
    for label in ["train", "test", "valid"]:
        raw_path = os.path.join("data", f"{label}23k.json")
        path = os.path.join("data", f"{label}23k_processed.json")
        num_process(raw_path, path)
        nlp_process("C:/Software/StanfordNLP", path, path)
        equ_process(path, path)
