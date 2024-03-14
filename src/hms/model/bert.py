# -*- coding:utf-8 -*-

import torch
from torch import nn

class BERTEmbed(nn.Module):
    def __init__(self, bert_path):
        super(BERTEmbed, self).__init__()
        from transformers import BertModel, BertTokenizerFast
        self.bert = BertModel.from_pretrained(bert_path, output_hidden_states=False, add_pooling_layer=False)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.segment_sep = ' '
        self.num_placeholder = 'n'
        self.embed_dim = self.bert.config.hidden_size
        self.freeze()
        return

    def freeze(self):
        for p in self.bert.parameters():
            p.requires_grad = False
        return
    
    def train(self, mode=True):
        self.bert.eval()
        return self
    
    def align_word_tokens(self, word_list, offset_mapping):
        # token in current word: token_start < word_end
        # token in following word: token_start >= word_end
        word_end_list = []
        acc_word_start = 0
        for word in word_list:
            word_end_list.append(acc_word_start + len(word))
            acc_word_start += len(word) + len(self.segment_sep)
        word_token_map = []
        temp_word_token = []
        word_idx = 0
        word_end = word_end_list[word_idx]
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            # token_end == 0: special token <cls> <sep> <pad>
            if token_end > 0:
                while word_idx < len(word_end_list) and token_start >= word_end:
                    word_token_map.append(temp_word_token)
                    temp_word_token = []
                    word_idx += 1
                    if word_idx < len(word_end_list):
                        word_end = word_end_list[word_idx]
                if word_idx < len(word_end_list):
                    temp_word_token.append(token_idx)
                else:
                    break
        if len(temp_word_token) > 0:
            word_token_map.append(temp_word_token)
        if len(word_token_map) > len(word_end_list):
            word_token_map = word_token_map[:len(word_end_list)]
        elif len(word_token_map) < len(word_end_list):
            word_token_map += [[]] * (len(word_end_list) - len(word_token_map))
        return word_token_map
    
    def encode_words(self, token_embeds, word_token_map, pad_embed, max_length):
        word_embeds = []
        for token_group in word_token_map:
            if len(token_group) == 0:
                word_embeds.append(pad_embed)
            else:
                word_embeds.append(token_embeds[token_group].mean(dim=0))
        if len(word_embeds) < max_length:
            word_embeds += [pad_embed] * (max_length - len(word_embeds))
        word_embeds = torch.stack(word_embeds, dim=0)
        return word_embeds

    def forward(self, batch_seq_id, batch_words):
        max_length = batch_seq_id.size(-1)
        batch_words = [[self.num_placeholder if w == "NUM_token" else w for w in words] for words in batch_words]
        texts = [self.segment_sep.join(words).lower() for words in batch_words]
        tokenized = self.tokenizer(texts, padding="longest", return_tensors="pt", return_offsets_mapping=True)
        offset_mappings = tokenized.pop("offset_mapping").tolist()
        word_token_maps = [self.align_word_tokens(word_list, offset_mapping) for word_list, offset_mapping in zip(batch_words, offset_mappings)]

        tokenized = tokenized.to(self.bert.device)
        embeds = self.bert(**tokenized).last_hidden_state.detach()
        pad_embed = torch.zeros(embeds.size(-1), device=embeds.device)
        batch_word_embeds = torch.stack([self.encode_words(embed, word_token_map, pad_embed, max_length) for embed, word_token_map in zip(embeds, word_token_maps)], dim=0)
        return batch_word_embeds
