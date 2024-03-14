# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

def get_mask(encode_lengths, pad_length):
    # 1 means pad, 0 means valid
    use_cuda = encode_lengths.is_cuda
    index = torch.arange(pad_length)
    if use_cuda:
        index = index.cuda()
    mask = (index.unsqueeze(0) >= encode_lengths.unsqueeze(-1)).byte()
    return mask

def pad_empty_mask(mask):
    # 1 means pad, 0 means valid
    # save one position for full padding span to prevent nan in softmax
    # invalid value in full padding span will be ignored in span level attention
    mask[(mask == 0).long().sum(dim=-1) == 0, 0] = 0
    return mask

class Attention(nn.Module):
    def __init__(self, hidden_dim, mix=True, fn=False, dropout=None):
        super(Attention, self).__init__()
        self.mix = mix
        self.fn = fn

        if self.fn:
            self.linear_out = nn.Linear(hidden_dim*2, hidden_dim)
        self.w = nn.Linear(hidden_dim*2, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        return

    def forward(self, query, context, mask=None):
        # query/context: batch_size * seq_len * hidden_dim
        # mask: batch_size * seq_len
        batch_size, query_size, _ = query.size()
        context_size = context.size(1)
        # batch_size * query_size * context_size * hidden_dim
        qc_query = query.unsqueeze(2).expand(-1, -1, context_size, -1)
        qc_context = context.unsqueeze(1).expand(-1, query_size, -1, -1)
        score_hidden = torch.cat((qc_query, qc_context), dim=-1)
        score_hidden = F.leaky_relu(self.w(score_hidden))
        score = self.score(score_hidden).view(batch_size, query_size, context_size)

        if mask is not None:
            if isinstance(mask, tuple):
                mask, true_mask = mask
            else:
                mask = mask.unsqueeze(1).expand(-1, query_size, -1)
                true_mask = None
            score.masked_fill_(mask==1, -float('inf'))
        attn = F.softmax(score, dim=-1)
        if mask is not None and true_mask is not None:
            attn = attn.masked_fill(true_mask==1, 0)
        if self.dropout is not None:
            attn = self.dropout(attn)

        if self.mix:
            # (b, q, c) * (b, c, d) -> (b, q, d)
            attn_output = torch.bmm(attn, context)
        else:
            attn_output = None

        if self.fn:
            combined = torch.cat((attn_output, query), dim=2)
            attn_output = F.leaky_relu(self.linear_out(combined))
        # attn_output: (b, q, d)
        # attn  : (b, q, c)
        return attn_output, attn

class HierarchicalAttention(nn.Module):
    def __init__(self, dim):
        super(HierarchicalAttention, self).__init__()
        self.span_attn = Attention(dim, mix=False, fn=False)
        self.word_attn = Attention(dim, mix=True, fn=False)
        return

    def forward(self, query, span_context, word_contexts, span_mask=None, word_masks=None):
        # query: batch_size * query_size * hidden_dim
        # span_context: batch_size * span_size * hidden_dim
        # word_contexts: (batch_size * seq_len * hidden_dim) * span_size
        batch_size, query_size, _ = query.size()
        _, span_size, hidden_dim = span_context.size()
        
        # batch_size * query_size * span_size
        _, span_attn = self.span_attn(query, span_context, span_mask)
        word_outputs = []
        for word_context, word_mask in zip(word_contexts, word_masks):
            # batch_size * query_size * hidden_dim
            word_output, _ = self.word_attn(query, word_context, word_mask)
            word_outputs.append(word_output.unsqueeze(-2))
        
        # batch_size * query_size * span_size * hidden_dim
        word_output = torch.cat(word_outputs, dim=-2)
        span_context = span_context.unsqueeze(1).expand(-1, query_size, -1, -1)
        # batch_size * query_size * hidden_dim
        # (b, q, s), (b, q, s, h) => (b, q, h)
        # (b, q, s) => (b*q, 1, s), (b, q, s, h) => (b*q, s, h)
        # (b*q, 1, s) * (b*q, s, h) => (b*q, 1, h) => (b, q, h)
        attn_output = torch.bmm(span_attn.view(-1, 1, span_size), (span_context + word_output).view(-1, span_size, hidden_dim)).view(batch_size, query_size, hidden_dim)
        return attn_output
