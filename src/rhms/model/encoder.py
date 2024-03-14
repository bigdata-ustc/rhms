# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

from .attention import get_mask, pad_empty_mask, Attention
from .bert import BERTEmbed

class PositionalEncoder(nn.Module):
    def __init__(self, pos_size, dim):
        super(PositionalEncoder, self).__init__()
        self.pos_size = pos_size
        pe = torch.randn(pos_size, dim)
        self.pe = nn.Parameter(pe)
        return
    
    def forward(self, input):
        # pe: seq_len * dim
        seq_len = input.size(1)
        if seq_len <= self.pos_size:
            pe = self.pe[:seq_len]
        else:
            pad = self.pe[-1].unsqueeze(0).expand(seq_len - self.pos_size, -1)
            pe = torch.cat((self.pe, pad), dim=0)
        # input: batch_size * seq_len * dim
        output = input + pe
        return output

class TreeEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(TreeEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        # merge subtree/word node
        self.to_parent = Attention(hidden_dim, mix=True, fn=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        return
    
    def dependency_encode(self, input, tree):
        # input: (1, seq_len, dim)
        # tree: (node, children, relations)
        # only support batch_size = 1
        word, children, relations = tree
        word_vector = input[:, word]
        if len(children) == 0:
            vector = word_vector
        else:
            children_vector = [self.dependency_encode(input, child).unsqueeze(1) for child in children]
            # (1, children_len, dim)
            children_vector = torch.cat(children_vector, dim=1)
            # (children_len) => (1, children_len) => (1, children_len, dim)
            query = word_vector.unsqueeze(1)
            query = self.dropout(query)
            children_vector = self.dropout(children_vector)
            vector = self.to_parent(query, children_vector)[0].squeeze(1)
        # (1, dim)
        return vector
    
    def batch_dependency_encode(self, batch_input, batch_tree):
        # batch_input: (batch_size, seq_len, dim)
        # batch_tree: [(node, children, relations)]
        pad_hidden = None
        batch_hidden = []
        for input, tree in zip(batch_input, batch_tree):
            if tree is not None:
                # (1, seq_len, dim)
                input = input.unsqueeze(0)
                # (1, dim)
                hidden = self.dependency_encode(input, tree)
            else:
                # lazy initialization
                if pad_hidden is None:
                    pad_hidden = torch.zeros(1, self.hidden_dim)
                    if batch_input.is_cuda:
                        pad_hidden = pad_hidden.cuda()
                hidden = pad_hidden
            batch_hidden.append(hidden)
        # (batch_size, dim)
        batch_hidden = torch.cat(batch_hidden, dim=0)
        return batch_hidden
    
    def get_encode_order(self, tree, all_steps):
        # tree: [(word, children, relations)], children: [(word, children, relations)]
        # all_steps: [(order_index, (word, children_word, relations))], order_index starts from 0
        word, children, relations = tree
        if len(children) == 0:
            order_index = -1
        else:
            children_word = [child_word for child_word, _, _ in children]
            children_order_index = [self.get_encode_order(child, all_steps) for child in children]
            max_child_order_index = max(children_order_index)
            if max_child_order_index == -1:
                order_index = 0
            else:
                order_index = max_child_order_index + 1
            step = [word, children_word, relations]
            all_steps.append((order_index, step))
        return order_index
    
    def get_serial_steps(self, tree, data_index):
        # tree: [(word, children, relations)], children: [(word, children, relations)]
        # ret: [serial_step], serial_step: [parallel_step], parallel_step: (data_index, (word, children_word, relations))
        all_steps = []
        self.get_encode_order(tree, all_steps)
        if len(all_steps) > 0:
            max_order_index = max([order_index for order_index, step in all_steps])
            serial_steps = [[(data_index, step) for step_order_index, step in all_steps if step_order_index == order_index] for order_index in range(max_order_index + 1)]
        else:
            serial_steps = []
        return serial_steps
    
    def get_batch_serial_steps(self, batch_tree):
        # [[(batch_index, (word, children_word, relations))]]
        # batch
        batch_serial_steps = []
        for batch_index, tree in enumerate(batch_tree):
            if tree is not None:
                serial_steps = self.get_serial_steps(tree, batch_index)
                for step_index, serial_step in enumerate(serial_steps):
                    if step_index >= len(batch_serial_steps):
                        batch_serial_steps.append([])
                    batch_serial_steps[step_index].extend(serial_step)
        # pad
        for serial_step in batch_serial_steps:
            max_children_size = max(len(children_word)  for _, (_, children_word, _) in serial_step)
            for _, parallel_step in serial_step:
                # word, children_word, relations
                children_word = parallel_step[1]
                pad_size = max_children_size - len(children_word)
                pad_children_word = children_word + [-1] * pad_size
                parallel_step[1] = pad_children_word
        return batch_serial_steps
        
    def fast_dependency_encode(self, batch_input, batch_tree):
        # batch_input: (batch_size, seq_len, dim)
        # batch_tree: [(node, children, relations)]
        use_cuda = batch_input.is_cuda
        # root word index, -1 means None
        root_index = [tree[0] if tree is not None else -1 for tree in batch_tree]
        # store all encoded and unencoded vector to avoid overwrite of gradient
        # dim
        vector_stack = [[vec for vec in seq_vec] for seq_vec in batch_input]
        pad_hidden = None
        for batch_idx, word_idx in enumerate(root_index):
            if word_idx == -1:
                # lazy initialization
                if pad_hidden is None:
                    pad_hidden = torch.zeros(self.hidden_dim)
                    if use_cuda:
                        pad_hidden = pad_hidden.cuda()
                vector_stack[batch_idx][word_idx] = pad_hidden
        
        # batch parallelizable encode steps
        serial_steps = self.get_batch_serial_steps(batch_tree)
        for parallel_batch in serial_steps:
            # [(batch_index, (word, children_word, relations))]
            # prepare data
            query = [vector_stack[batch_index][word_index] for batch_index, (word_index, _, _) in parallel_batch]
            context = [[vector_stack[batch_index][child_index] for child_index in children_index] for batch_index, (_, children_index, _) in parallel_batch]
            mask = [children_index for _, (_, children_index, _) in parallel_batch]
            # step_size * dim
            query = torch.stack(query, dim=0)
            # step_size * children_len * dim
            context = torch.stack([torch.stack(children, dim=0) for children in context], dim=0)
            # step_size * children_len
            mask = torch.LongTensor(mask)
            if use_cuda:
                mask = mask.cuda()
            # 0 means valid, 1 means pad
            mask = (mask == -1).byte()
            
            # encode
            query = query.unsqueeze(1)
            query = self.dropout(query)
            context = self.dropout(context)
            # step_size * dim
            vector = self.to_parent(query, context, mask)[0].squeeze(1)
            
            # update vector_stack
            for step_index, (batch_index, (word_index, _, _)) in enumerate(parallel_batch):
                vector_stack[batch_index][word_index] = vector[step_index]
        
        # fetch root vector
        batch_hidden = [vector_stack[batch_idx][word_idx] for batch_idx, word_idx in enumerate(root_index)]
        # (batch_size, dim)
        batch_hidden = torch.stack(batch_hidden, dim=0)
        return batch_hidden

    def forward(self, span_batch_input, span_batch_tree):
        span_batch_hidden = []
        for batch_input, batch_tree in zip(span_batch_input, span_batch_tree):
            # batch_hidden = self.batch_dependency_encode(batch_input, batch_tree)
            batch_hidden = self.fast_dependency_encode(batch_input, batch_tree)
            span_batch_hidden.append(batch_hidden)
        return span_batch_hidden

class GATModule(nn.Module):
    def __init__(self, hidden_dim, hop, head, dropout):
        super(GATModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.hop = hop
        self.head = head
        attn_dim = hidden_dim // head

        self.proj_fn = nn.Linear(hidden_dim, hidden_dim)        
        self.ws = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_dim, attn_dim) for _ in range(head)]) for _ in range(hop)])
        self.gats = nn.ModuleList([nn.ModuleList([Attention(attn_dim, mix=True, fn=True, dropout=dropout) for _ in range(head)]) for _ in range(hop)])
        self.ff = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        return
    
    def forward(self, span_hidden, span_output, neighbor_span_output, span_mask, neighbor_span_mask, graph_map):
        # span_hidden: batch_size * dim
        # span_output: batch_size * span_len * dim
        # neighbor_span_outputs: neighbor_size * span_len * dim
        # span_mask: batch_size * span_len
        # neighbor_span_mask: neighbor_size * span_len
        # graph_map: batch_size * max_neighbor_len, -1 means pad
        use_cuda = span_output.is_cuda
        batch_size = span_hidden.size(0)

        # mean pool
        # size * dim
        # (b, l, d) -> (b, d)
        span_output = self.dropout(span_output)
        span_output = self.proj_fn(span_output)
        b_hidden = span_output.masked_fill(span_mask.unsqueeze(-1) == 1, 0).sum(dim=1) / (span_mask != 1).sum(dim=1).unsqueeze(-1)
        b_hidden = F.leaky_relu(b_hidden)
        # batch_size, whether each sample has no neighbor
        batch_empty_mask = (graph_map[:batch_size] != -1).long().sum(dim=-1) == 0

        if (graph_map != -1).sum().item() > 0 and neighbor_span_output is not None:
            neighbor_span_output = self.dropout(neighbor_span_output)
            neighbor_span_output = self.proj_fn(neighbor_span_output)
            n_hidden = neighbor_span_output.masked_fill(neighbor_span_mask.unsqueeze(-1) == 1, 0).sum(dim=1) / (neighbor_span_mask != 1).sum(dim=1).unsqueeze(-1)
            n_hidden = F.leaky_relu(n_hidden)
            # all_size * dim
            # neighbor_pos: batch_data + neighbor_data
            n_hidden = torch.cat((b_hidden, n_hidden), dim=0)
            
            # mask
            # batch_size * max_neighbor_len
            # 1 means pad, 0 means valid
            mask = (graph_map == -1).byte()
            pad_mask = pad_empty_mask(mask, clone=True)
            # batch_size, whether each batch is all empty
            empty_mask = (graph_map != -1).long().sum(dim=-1) == 0
            # frontier neighbors only used in fist hop for the sub-frontier neighbors
            skip_hidden = n_hidden[graph_map.size(0):]
            
            c_hidden = n_hidden
            for hop_idx in range(self.hop):
                if hop_idx == self.hop - 1:
                    # only batch hidden used in last hop
                    c_graph_map = graph_map[:batch_size]
                else:
                    c_graph_map = graph_map
                node_size = c_graph_map.size(0)
                c_pad_mask = pad_mask[:node_size]
                c_empty_mask = empty_mask[:node_size]
                
                c_hidden = self.dropout(c_hidden)
                c_hiddens = []
                for head_idx in range(self.head):
                    ch_hidden = self.ws[hop_idx][head_idx](c_hidden)
                    ch_hidden = F.leaky_relu(ch_hidden)
                    node_hidden = ch_hidden[:node_size]
                    # batch_size * max_neighbor_len * dim
                    context = ch_hidden[c_graph_map]
                    # batch_size * 1 * dim
                    query = node_hidden.unsqueeze(1)
                    # batch_size * dim
                    ch_hidden = self.gats[hop_idx][head_idx](query, context, c_pad_mask)[0].squeeze(1)
                    c_hiddens.append(ch_hidden)
                c_hidden = torch.cat(c_hiddens, dim=-1)
                # node without edges
                c_hidden.masked_fill_(c_empty_mask.unsqueeze(-1), 0)
                # only neighbors of batch data required for last hop
                if hop_idx < self.hop - 2:
                    c_hidden = torch.cat((c_hidden, skip_hidden), dim=0)
        else:
            c_hidden = torch.zeros(batch_size, self.hidden_dim)
            if use_cuda:
                c_hidden = c_hidden.cuda()
        
        # samples without neighbors use self as "neighbor" hidden
        c_hidden[batch_empty_mask] = b_hidden[batch_empty_mask]
        hidden = torch.cat((span_hidden, c_hidden), dim=-1)
        span_hidden = F.leaky_relu(self.ff(hidden))
        return span_hidden

class Encoder(nn.Module):
    def __init__(self, embed_model, hidden_dim, span_size, hop, head, dropout, bert_path=None):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.hop = hop

        if bert_path is None:
            self.embedding = embed_model
            embed_dim = self.embedding.embedding_dim
            self.use_bert = False
        else:
            self.embedding = BERTEmbed(bert_path)
            embed_dim = self.embedding.embed_dim
            self.use_bert = True
        # word encoding
        self.word_rnn = nn.GRU(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        # span encoding
        # span sequence
        self.span_attn = Attention(hidden_dim, mix=True, fn=True, dropout=dropout)
        self.pos_enc = PositionalEncoder(span_size, hidden_dim)
        # tree
        self.dep_enc = TreeEncoder(hidden_dim, dropout=dropout)
        if self.hop > 0:
            self.gat_module = GATModule(hidden_dim, hop, head, dropout=dropout)
        return
    
    def fetch_seq(self, batch_data, batch_mask):
        use_cuda = batch_data.is_cuda
        batch_seq = []
        batch_pad = []
        batch_len = []
        for data, mask in zip(batch_data, batch_mask):
            seq = data[mask]
            batch_seq.append(seq)
            batch_len.append(seq.size(0))
            if (~mask).sum().item() > 0:
                pad = data[~mask][0].view(-1)
            else:
                pad = None
            batch_pad.append(pad)
        max_seq_len = max(seq.size(0) for seq in batch_seq)
        for batch_idx, (seq, pad) in enumerate(zip(batch_seq, batch_pad)):
            seq_len = seq.size(0)
            if seq_len < max_seq_len:
                batch_seq[batch_idx] = torch.cat((seq, pad.repeat(max_seq_len - seq_len)), dim=0)
        batch_seq = torch.stack(batch_seq, dim=0)
        batch_len = torch.tensor(batch_len)
        if use_cuda:
            batch_len = batch_len.cuda()
        return batch_seq, batch_len

    def combine_fwd_bwd(self, output, hidden):
        # combine forward and backward LSTM
        if output is not None:
            output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # (num_layers * num_directions, batch, hidden_dim).view(num_layers, num_directions, batch, hidden_dim)
        if hidden is not None:
            hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        return output, hidden
    
    def seq_encode(self, inputs, input_lengths, words):
        # [span1, span2, ...] => seq
        # [batch_size * seq_len] => batch_size * seq_len
        span_lens = [span.size(1) for span in inputs]
        span_masks = [get_mask(span_len, span.size(1)) == 0 for span, span_len in zip(inputs, input_lengths)]
        batch_data = torch.cat(inputs, dim=1)
        batch_mask = torch.cat(span_masks, dim=1)
        batch_seq, batch_len = self.fetch_seq(batch_data, batch_mask)
        seq_mask = get_mask(batch_len, batch_seq.size(1)) == 0

        # seq => seq_vec
        # word encoding
        if self.use_bert:
            embedded = self.embedding(batch_seq, words)
        else:
            embedded = self.embedding(batch_seq)
        embedded = self.dropout(embedded)
        # output: (batch_size, seq_len, 2*dim) => (batch_size, seq_len, 2, dim)
        # hidden: (layer*2, batch_size, dim) => (layer, 2, batch_size, dim)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, batch_len.cpu(), batch_first=True, enforce_sorted=False)
        seq_output, _ = self.word_rnn(embedded, None)
        seq_output, _ = nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)
        # batch_size * seq_len * dim
        seq_output, _ = self.combine_fwd_bwd(seq_output, None)

        # seq_vec => [span1_vec, span2_vec, ...]
        # batch_size * seq_len * dim => [batch_size * seq_len * dim]
        batch_size, padded_len = batch_data.size()
        hidden_dim = seq_output.size(-1)
        padded_output = torch.zeros(batch_size, padded_len, hidden_dim)
        if seq_output.is_cuda:
            padded_output = padded_output.cuda()
        padded_output[batch_mask] = seq_output[seq_mask]
        batch_output = padded_output.split(span_lens, dim=1)
        return batch_output

    def encode(self, inputs, input_lengths, span_length, trees, words):
        use_cuda = span_length.is_cuda
        
        # [batch_size * seq_len * dim]
        word_outputs = self.seq_encode(inputs, input_lengths, words)
        # dependency encoding
        # [batch_size * dim]
        span_inputs = self.dep_enc(word_outputs, trees)
        
        # span encoding
        # batch_size * span_len * dim
        span_input = torch.stack(span_inputs, dim=1)
        span_mask = get_mask(span_length, span_input.size(1))
        # mask self
        span_size = span_mask.size(1)
        span_mask_ignore_self = torch.stack([span_mask.clone() for _ in range(span_size)], dim=1)
        self_indice = torch.arange(span_size)
        if use_cuda:
            self_indice = self_indice.cuda()
        span_mask_ignore_self[:, self_indice, self_indice] = 1
        # batch with only one span
        one_span_batch = (span_mask != 1).sum(dim=-1) == 1
        if one_span_batch.sum().item() > 0:
            pad_span_mask_ignore_self = span_mask_ignore_self.clone()
            pad_span_mask_ignore_self[one_span_batch, :, 0] = 0
        else:
            pad_span_mask_ignore_self = span_mask_ignore_self
            span_mask_ignore_self = None
        mask = (pad_span_mask_ignore_self, span_mask_ignore_self)

        span_output = self.pos_enc(span_input)
        span_output = self.dropout(span_output)
        span_output, _ = self.span_attn(span_output, span_output, mask)
        # batch_size * span_len * dim => batch_size * dim
        span_hidden = span_output.masked_fill(span_mask.unsqueeze(-1) == 1, 0).sum(dim=1) / (span_mask != 1).sum(dim=1).unsqueeze(-1)
        return span_output, word_outputs, span_hidden, span_mask
    
    def forward(self, inputs, input_lengths, span_length, trees, words,
                neighbor_inputs=None, neighbor_input_lengths=None, neighbor_span_length=None, neighbor_trees=None, neighbor_words=None,
                graph_map=None, warm_training=False):
        span_output, word_outputs, span_hidden, span_mask = self.encode(inputs, input_lengths, span_length, trees, words)
        # gat module for question graph
        if self.hop > 0 and (not warm_training):
            if neighbor_inputs is not None and neighbor_span_length.size(0) > 0:
                neighbor_span_output, _, _, neighbor_span_mask = self.encode(neighbor_inputs, neighbor_input_lengths, neighbor_span_length, neighbor_trees, neighbor_words)
            else:
                neighbor_span_output, neighbor_span_mask = None, None
            span_hidden = self.gat_module(span_hidden, span_output, neighbor_span_output, span_mask, neighbor_span_mask, graph_map)
        return (span_output, word_outputs), span_hidden
