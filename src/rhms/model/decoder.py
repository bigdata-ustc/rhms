# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from .attention import get_mask, pad_empty_mask, HierarchicalAttention

class GateNN(nn.Module):
    def __init__(self, hidden_dim, input1_dim, input2_dim=0, dropout=0.5, single_layer=False):
        super(GateNN, self).__init__()
        self.single_layer = single_layer
        self.hidden_l1 = nn.Linear(input1_dim+hidden_dim, hidden_dim)
        self.gate_l1 = nn.Linear(input1_dim+hidden_dim, hidden_dim)
        if not single_layer:
            self.dropout = nn.Dropout(p=dropout)
            self.hidden_l2 = nn.Linear(input2_dim+hidden_dim, hidden_dim)
            self.gate_l2 = nn.Linear(input2_dim+hidden_dim, hidden_dim)
        return
    
    def forward(self, hidden, input1, input2=None):
        input1 = torch.cat((hidden, input1), dim=-1)
        h = torch.tanh(self.hidden_l1(input1))
        g = torch.sigmoid(self.gate_l1(input1))
        h = h * g
        if not self.single_layer:
            h1 = self.dropout(h)
            if input2 is not None:
                input2 = torch.cat((h1, input2), dim=-1)
            else:
                input2 = h1
            h = torch.tanh(self.hidden_l2(input2))
            g = torch.sigmoid(self.gate_l2(input2))
            h = h * g
        return h

class ScoreModel(nn.Module):
    def __init__(self, hidden_dim):
        super(ScoreModel, self).__init__()
        self.w = nn.Linear(hidden_dim * 3, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        return
    
    def forward(self, hidden, context, token_embeddings):
        # hidden/context: batch_size * hidden_dim
        # token_embeddings: batch_size * class_size * hidden_dim
        _, class_size, _ = token_embeddings.size()
        hc = torch.cat((hidden, context), dim=-1)
        # (b, c, h)
        hc = hc.unsqueeze(1).expand(-1, class_size, -1)
        hidden = torch.cat((hc, token_embeddings), dim=-1)
        hidden = F.leaky_relu(self.w(hidden))
        score = self.score(hidden).squeeze(-1)
        return score

class PredictModel(nn.Module):
    def __init__(self, hidden_dim, class_size, op_size, dropout, use_cuda=True):
        super(PredictModel, self).__init__()
        self.class_size = class_size
        self.use_cuda = use_cuda

        self.dropout = nn.Dropout(p=dropout)
        self.attn = HierarchicalAttention(hidden_dim)
        
        self.score_pointer = ScoreModel(hidden_dim)
        self.score_gen = ScoreModel(hidden_dim)
        self.score_op = nn.Linear(hidden_dim*2, op_size)
        self.gen_prob = nn.Linear(hidden_dim*2, 1)
        return
    
    def score_pn(self, hidden, context, embedding_masks, finished_mask):
        # embedding: batch_size * pointer_size * hidden_dim
        # mask: batch_size * pointer_size
        (pointer_embedding, pointer_mask), const_embedding, op_embedding, _ = embedding_masks
        hidden = self.dropout(hidden)
        context = self.dropout(context)
        # pointer
        pointer_embedding = self.dropout(pointer_embedding)
        # batch_size * symbol_size
        pointer_score = self.score_pointer(hidden, context, pointer_embedding)
        pointer_score.masked_fill_(pointer_mask, -float('inf'))
        
        # generator
        # pad + op + const
        const_embedding = self.dropout(const_embedding)
        const_score = self.score_gen(hidden, context, const_embedding)
        op_score = self.score_op(torch.cat((hidden, context), dim=-1))
        # batch_size * generator_size
        generator_score = torch.cat((op_score, const_score), dim=-1)

        # pad
        # finished_mask = 0 => unfinished => pad_score = -inf
        # finished_mask = 1 => finished => pad_score = 0, others = -inf
        pad_score = torch.zeros(finished_mask.size())
        if self.use_cuda:
            pad_score = pad_score.cuda()
        pad_score.masked_fill_(finished_mask == 0, -float('inf'))
        pointer_score.masked_fill_(finished_mask == 1, -float('inf'))
        generator_score.masked_fill_(finished_mask == 1, -float('inf'))
        # pad + op + const
        generator_score = torch.cat((pad_score, generator_score), dim=-1)        
        return pointer_score, generator_score

    def forward(self, node_hidden, encoder_outputs, masks, embedding_masks, finished_mask):
        span_output, word_outputs = encoder_outputs
        span_mask, word_masks = masks
        dp_hidden = self.dropout(node_hidden)
        output_attn = self.attn(dp_hidden.unsqueeze(1), span_output, word_outputs, span_mask, word_masks)
        context = output_attn.squeeze(1)

        # log(f(softmax(x)))
        finished_mask = finished_mask.unsqueeze(-1)
        pointer_score, generator_score = self.score_pn(node_hidden, context, embedding_masks, finished_mask)
        hc = torch.cat((node_hidden, context), dim=-1)
        hc = self.dropout(hc)

        # batch_size * 1, batch_size * class_size
        gen_score = self.gen_prob(hc)
        # finished_mask = 1 => finished => PAD_token => gen_score = inf
        gen_score.masked_fill_(finished_mask == 1, float('inf'))
        # log(sigmoid * softmax) = log_sigmoid + log_softmax
        log_generator = F.logsigmoid(gen_score) + F.log_softmax(generator_score, dim=-1)
        # log((1 - sigmoid) * softmax) = log(1 - sigmoid) + log_softmax = log_sigmoid(-x) + log_softmax
        log_pointer = F.logsigmoid(-gen_score) + F.log_softmax(pointer_score, dim=-1)
        log_pointer.masked_fill_(finished_mask == 1, -float('inf'))
        # pad + op + const + pointer + empty_pointer
        log_score = torch.cat((log_generator, log_pointer), dim=-1)
        pad_empty_pointer = torch.ones(log_score.size(0), self.class_size - log_score.size(-1)) * (-float('inf'))
        if self.use_cuda:
            pad_empty_pointer = pad_empty_pointer.cuda()
        output = torch.cat((log_score, pad_empty_pointer), dim=-1)
        return output, context

class TreeEmbeddingNode:
    def __init__(self, embedding, terminal):
        self.embedding = embedding
        self.terminal = terminal
        return

class TreeEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim, op_set, dropout):
        super(TreeEmbeddingModel, self).__init__()
        self.op_set = op_set
        self.dropout = nn.Dropout(p=dropout)
        self.combine = GateNN(hidden_dim, hidden_dim*2, dropout=dropout, single_layer=True)
        return
    
    def merge(self, op_embedding, left_embedding, right_embedding):
        te_input = torch.cat((left_embedding, right_embedding), dim=-1)
        te_input = self.dropout(te_input)
        op_embedding = self.dropout(op_embedding)
        tree_embed = self.combine(op_embedding, te_input)
        return tree_embed
    
    def serial_forward(self, class_embedding, tree_stacks, embed_node_index):
    # def forward(self, class_embedding, tree_stacks, embed_node_index):
        # embed_node_index: batch_size
        use_cuda = embed_node_index.is_cuda
        batch_index = torch.arange(embed_node_index.size(0))
        if use_cuda:
            batch_index = batch_index.cuda()
        labels_embedding = class_embedding[batch_index, embed_node_index]
        for node_label, tree_stack, label_embedding in zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding):
            # operations
            if node_label in self.op_set:
                tree_node = TreeEmbeddingNode(label_embedding, terminal=False)
            # numbers
            else:
                right_embedding = label_embedding
                # on right tree => merge
                while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
                    left_embedding = tree_stack.pop().embedding
                    op_embedding = tree_stack.pop().embedding
                    right_embedding = self.merge(op_embedding, left_embedding, right_embedding)
                tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
            tree_stack.append(tree_node)
        return labels_embedding

    def get_merge_embeddings(self, tree_stack):
        left_embeddings = []
        op_embeddings = []
        # on right tree => merge
        while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
            left_embedding = tree_stack.pop().embedding
            op_embedding = tree_stack.pop().embedding
            left_embeddings.append(left_embedding)
            op_embeddings.append(op_embedding)
        return left_embeddings, op_embeddings
    
    # def fast_forward(self, class_embedding, tree_stacks, embed_node_index):
    def forward(self, class_embedding, tree_stacks, embed_node_index):
        # embed_node_index: batch_size
        use_cuda = embed_node_index.is_cuda
        batch_index = torch.arange(embed_node_index.size(0))
        if use_cuda:
            batch_index = batch_index.cuda()
        labels_embedding = class_embedding[batch_index, embed_node_index]
        merge_batch = []
        right_embeddings = []
        all_left_embeddings = []
        all_op_embeddings = []
        batch_step_size = []
        # get merge steps
        for batch_index, (node_label, tree_stack, label_embedding) in enumerate(zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding)):
            # operations
            if node_label in self.op_set:
                tree_node = TreeEmbeddingNode(label_embedding, terminal=False)
                tree_stack.append(tree_node)    # no need to merge
            # numbers
            else:
                right_embedding = label_embedding
                left_embeddings, op_embeddings = self.get_merge_embeddings(tree_stack)
                current_step_size = len(left_embeddings)
                if current_step_size > 0:
                    merge_batch.append(batch_index)
                    right_embeddings.append(right_embedding)
                    all_left_embeddings.append(left_embeddings)
                    all_op_embeddings.append(op_embeddings)
                    batch_step_size.append(current_step_size)
                else:
                    tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
                    tree_stack.append(tree_node)    # no need to merge
        # data need to merge
        # batch parallel steps
        # sort all data to merge by step_count from high to low
        if len(merge_batch) > 0:
            embed_idx_size = list(enumerate(batch_step_size))
            embed_idx_size.sort(key=lambda idx_size: idx_size[1], reverse=True)
            embed_idx, batch_step_size = list(zip(*embed_idx_size))
            merge_batch = [merge_batch[idx] for idx in embed_idx]
            right_embeddings = [right_embeddings[idx] for idx in embed_idx]
            # convert batch_data to step_data
            # [batch1_embeddings, batch2_embeddings, ...]
            # batch1_embeddings: [step1_embedding, step2_embedding, ...]
            all_left_embeddings = [all_left_embeddings[idx] for idx in embed_idx]
            all_op_embeddings = [all_op_embeddings[idx] for idx in embed_idx]
            # [step1_embeddings, step2_embeddings, ...]
            # step1_embeddings: [batch1_embedding, batch2_embedding, ...]
            max_step_size = batch_step_size[0]
            # require batch_data in order by step_count from high to low
            serial_left_embeddings = [[batch_data[step_index] for batch_data in all_left_embeddings if step_index < len(batch_data)] for step_index in range(max_step_size)]
            serial_op_embeddings = [[batch_data[step_index] for batch_data in all_op_embeddings if step_index < len(batch_data)] for step_index in range(max_step_size)]
            step_batch_size = [len(batch_data) for batch_data in serial_left_embeddings]
            # batch merge
            right_embeddings = torch.stack(right_embeddings, dim=0)
            last_step_size = -1
            for size, left_embeddings, op_embeddings in zip(step_batch_size, serial_left_embeddings, serial_op_embeddings):
                # low step batch end merging, add merged embedding to tree_stack
                # require batch_data in order by step_count from high to low
                if last_step_size >= 0 and size != last_step_size:
                    end_size = last_step_size - size
                    for end_index in range(end_size):
                        end_index = size + end_index
                        batch_index = merge_batch[end_index]
                        right_embedding = right_embeddings[end_index]
                        tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
                        tree_stacks[batch_index].append(tree_node)
                last_step_size = size
                # high step batch continue merging
                right_embeddings = right_embeddings[:size]
                left_embeddings = torch.stack(left_embeddings, dim=0)
                op_embeddings = torch.stack(op_embeddings, dim=0)
                right_embeddings = self.merge(op_embeddings, left_embeddings, right_embeddings)
            # merged embedding for last step
            for end_index in range(last_step_size):
                batch_index = merge_batch[end_index]
                right_embedding = right_embeddings[end_index]
                tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
                tree_stacks[batch_index].append(tree_node)
        return labels_embedding 

class NodeEmbeddingNode:
    def __init__(self, node_hidden, node_context=None, label_embedding=None):
        self.node_hidden = node_hidden
        self.node_context = node_context
        self.label_embedding = label_embedding
        return

class DecomposeModel(nn.Module):
    def __init__(self, hidden_dim, dropout, use_cuda=True):
        super(DecomposeModel, self).__init__()
        self.pad_hidden = torch.zeros(hidden_dim)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.pad_hidden = self.pad_hidden.cuda()

        self.dropout = nn.Dropout(p=dropout)
        self.l_decompose = GateNN(hidden_dim, hidden_dim*2, 0, dropout=dropout, single_layer=False)
        self.r_decompose = GateNN(hidden_dim, hidden_dim*2, hidden_dim, dropout=dropout, single_layer=False)
        return
    
    def serial_forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
    # def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        children_hidden = []
        finished_mask = []
        for node_stack, tree_stack, node_context, label_embedding in zip(node_stacks, tree_stacks, nodes_context, labels_embedding):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden    # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)   # add context and label of parent for right child
                    l_input = torch.cat((node_context, label_embedding), dim=-1)
                    l_input = self.dropout(l_input)
                    node_hidden = self.dropout(node_hidden)
                    child_hidden = self.l_decompose(node_hidden, l_input, None)
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child
                # right
                else:
                    node_stack.pop()    # left child or last node, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding   # left tree
                        left_embedding = self.dropout(left_embedding)
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        r_input = self.dropout(r_input)
                        node_hidden = self.dropout(node_hidden)
                        child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
                        node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
                    # else finished decode
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = self.pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
                finished_mask.append(1)
            else:
                finished_mask.append(0)
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        finished_mask = torch.LongTensor(finished_mask)
        if self.use_cuda:
            finished_mask = finished_mask.cuda()
        return children_hidden, finished_mask
    
    # def fast_forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
    def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        hidden_flags = []
        left_batch_index = []
        left_node_hidden = []
        left_node_context = []
        left_label_embedding = []
        right_batch_index = []
        right_node_hidden = []
        right_node_context = []
        right_label_embedding = []
        right_left_embedding = []
        # batch left data and right data
        for batch_index, (node_stack, tree_stack, node_context, label_embedding) in enumerate(zip(node_stacks, tree_stacks, nodes_context, labels_embedding)):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden    # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)   # add context and label of parent for right child
                    left_batch_index.append(batch_index)
                    left_node_hidden.append(node_hidden)
                    left_node_context.append(node_context)
                    left_label_embedding.append(label_embedding)
                    index_in_left = len(left_batch_index) - 1
                    hidden_flags.append(('left', index_in_left))
                # right
                else:
                    node_stack.pop()    # left child or last node, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding   # left tree
                        right_batch_index.append(batch_index)
                        right_node_hidden.append(node_hidden)
                        right_node_context.append(node_context)
                        right_label_embedding.append(label_embedding)
                        right_left_embedding.append(left_embedding)
                        index_in_right = len(right_batch_index) - 1
                        hidden_flags.append(('right', index_in_right))
                    else:
                        # finished decode
                        hidden_flags.append(('none',))   # pad
            else:
                hidden_flags.append(('none',))   # pad
        # batch left decompose
        if len(left_batch_index) > 0:
            node_hidden = torch.stack(left_node_hidden, dim=0)
            node_context = torch.stack(left_node_context, dim=0)
            label_embedding = torch.stack(left_label_embedding, dim=0)
            l_input = self.dropout(torch.cat((node_context, label_embedding), dim=-1))
            node_hidden = self.dropout(node_hidden)
            left_child_hidden = self.l_decompose(node_hidden, l_input, None)
        # batch right decompose
        if len(right_batch_index) > 0:
            node_hidden = torch.stack(right_node_hidden, dim=0)
            node_context = torch.stack(right_node_context, dim=0)
            label_embedding = torch.stack(right_label_embedding, dim=0)
            left_embedding = torch.stack(right_left_embedding, dim=0)
            left_embedding = self.dropout(left_embedding)
            r_input = self.dropout(torch.cat((node_context, label_embedding), dim=-1))
            node_hidden = self.dropout(node_hidden)
            right_child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
        # post process
        # for left
        for child_hidden_index, batch_index in enumerate(left_batch_index):
            child_hidden = left_child_hidden[child_hidden_index]
            node_stacks[batch_index].append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child
        # for right
        for child_hidden_index, batch_index in enumerate(right_batch_index):
            child_hidden = right_child_hidden[child_hidden_index]
            node_stacks[batch_index].append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
        # for all after above
        children_hidden = []
        finished_mask = []
        for batch_index, node_stack, in enumerate(node_stacks):
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = self.pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
                finished_mask.append(1)
            else:
                group, index_in_group = hidden_flags[batch_index]
                if group == "left":
                    child_hidden = left_child_hidden[index_in_group]
                elif group == "right":
                    child_hidden = right_child_hidden[index_in_group]
                finished_mask.append(0)
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        finished_mask = torch.LongTensor(finished_mask)
        if self.use_cuda:
            finished_mask = finished_mask.cuda()
        return children_hidden, finished_mask

def copy_list(src_list):
    dst_list = [copy_list(item) if type(item) is list else item for item in src_list]
    return dst_list

class BeamNode:
    def __init__(self, score, nodes_hidden, finished_mask, node_stacks, tree_stacks, decoder_outputs_list, sequence_symbols_list):
        self.score = score
        self.nodes_hidden = nodes_hidden
        self.finished_mask = finished_mask
        self.node_stacks = node_stacks
        self.tree_stacks = tree_stacks
        self.decoder_outputs_list = decoder_outputs_list
        self.sequence_symbols_list = sequence_symbols_list
        return
    
    def copy(self):
        node = BeamNode(
            self.score,
            self.nodes_hidden,
            self.finished_mask,
            copy_list(self.node_stacks),
            copy_list(self.tree_stacks),
            copy_list(self.decoder_outputs_list),
            copy_list(self.sequence_symbols_list)
        )
        return node

class Decoder(nn.Module):
    def __init__(self, embed_model, op_set, vocab_dict, class_list, hidden_dim, dropout, use_cuda=True):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        embed_dim = embed_model.embedding_dim
        class_size = len(class_list)

        self.get_predict_meta(class_list, vocab_dict, op_set)

        self.embed_model = embed_model
        # 128 => 512
        self.gen_embed_to_hidden = nn.Linear(embed_dim, hidden_dim)
        self.predict = PredictModel(hidden_dim, class_size, self.op_vocab.size(0), dropout=dropout, use_cuda=use_cuda)
        op_idx_set = set(i for i, symbol in enumerate(class_list) if symbol in op_set)
        self.tree_embedding = TreeEmbeddingModel(hidden_dim, op_idx_set, dropout=dropout)
        self.decompose = DecomposeModel(hidden_dim, dropout=dropout, use_cuda=use_cuda)
        return

    def get_predict_meta(self, class_list, vocab_dict, op_set):
        # embed order: generator + pointer, with original order
        # used in predict_model, tree_embedding
        pointer_list = [token for token in class_list if 'temp_' in token]
        ctrl_list = [token for token in class_list if token not in pointer_list and "_token" in token]
        op_list = [token for token in class_list if token not in pointer_list and token in op_set]
        const_list = [token for token in class_list if token not in pointer_list and token not in ctrl_list and token not in op_list]
        embed_list = ctrl_list + op_list + const_list + pointer_list

        # pointer num index in class_list, for select only num pos from num_pos with op pos
        self.pointer_index = torch.LongTensor([class_list.index(token) for token in pointer_list])
        # generator symbol index in vocab, for generator symbol embedding
        self.ctrl_vocab = torch.LongTensor([vocab_dict[token] for token in ctrl_list])
        self.op_vocab = torch.LongTensor([vocab_dict[token] for token in op_list])
        self.const_vocab = torch.LongTensor([vocab_dict[token] for token in const_list])
        # class_index -> embed_index, for tree embedding
        # embed order -> class order, for predict_model output
        self.class_to_embed_index = torch.LongTensor([embed_list.index(token) for token in class_list])
        if self.use_cuda:
            self.pointer_index = self.pointer_index.cuda()
            self.ctrl_vocab = self.ctrl_vocab.cuda()
            self.op_vocab = self.op_vocab.cuda()
            self.const_vocab = self.const_vocab.cuda()
            self.class_to_embed_index = self.class_to_embed_index.cuda()
        return

    def get_pad_masks(self, encoder_outputs, input_lengths, span_length):
        span_output, word_outputs = encoder_outputs
        span_pad_length = span_output.size(1)
        word_pad_lengths = [word_output.size(1) for word_output in word_outputs]
        
        span_mask = get_mask(span_length, span_pad_length)
        word_masks = [pad_empty_mask(get_mask(input_length, word_pad_length), False) for input_length, word_pad_length in zip(input_lengths, word_pad_lengths)]
        masks = (span_mask, word_masks)
        return masks

    def get_pointer_meta(self, num_pos, sub_num_poses):
        batch_size = num_pos.size(0)
        pointer_num_pos = num_pos.index_select(dim=1, index=self.pointer_index)
        num_pos_occupied = pointer_num_pos.sum(dim=0) == -batch_size
        occupied_len = num_pos_occupied.size(-1)
        for i, elem in enumerate(reversed(num_pos_occupied.cpu().tolist())):
            if not elem:
                occupied_len = occupied_len - i
                break
        pointer_num_pos = pointer_num_pos[:, :occupied_len]
        # length of word_num_poses determined by span_num_pos
        sub_pointer_poses = [sub_num_pos.index_select(dim=1, index=self.pointer_index)[:, :occupied_len] for sub_num_pos in sub_num_poses]
        return pointer_num_pos, sub_pointer_poses

    def get_pointer_embedding(self, pointer_num_pos, encoder_outputs):
        # encoder_outputs: batch_size * seq_len * hidden_dim
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        batch_size, pointer_size = pointer_num_pos.size()
        batch_index = torch.arange(batch_size)
        if self.use_cuda:
            batch_index = batch_index.cuda()
        batch_index = batch_index.unsqueeze(1).expand(-1, pointer_size)
        # batch_size * pointer_len * hidden_dim
        pointer_embedding = encoder_outputs[batch_index, pointer_num_pos]
        # mask invalid pos -1
        pointer_embedding = pointer_embedding * (pointer_num_pos != -1).unsqueeze(-1)
        return pointer_embedding
    
    def get_pointer_mask(self, pointer_num_pos):
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        pointer_mask = pointer_num_pos == -1
        return pointer_mask
    
    def get_generator_embedding(self, batch_size):
        # generator_size * hidden_dim
        ctrl_embedding = self.gen_embed_to_hidden(self.embed_model(self.ctrl_vocab))
        op_embedding = self.gen_embed_to_hidden(self.embed_model(self.op_vocab))
        const_embedding = self.gen_embed_to_hidden(self.embed_model(self.const_vocab))
        generator_embedding = torch.cat((ctrl_embedding, op_embedding, const_embedding), dim=0)
        # batch_size * generator_size * hidden_dim
        op_embedding = op_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        const_embedding = const_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        generator_embedding = generator_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return const_embedding, op_embedding, generator_embedding
    
    def get_class_embedding_mask(self, num_pos, encoder_outputs):
        # embedding: batch_size * size * hidden_dim
        # mask: batch_size * size
        _, word_outputs = encoder_outputs
        span_num_pos, word_num_poses = num_pos
        const_embedding, op_embedding, generator_embedding = self.get_generator_embedding(span_num_pos.size(0))
        span_pointer_num_pos, word_pointer_num_poses = self.get_pointer_meta(span_num_pos, word_num_poses)
        pointer_mask = self.get_pointer_mask(span_pointer_num_pos)
        num_pointer_embeddings = []
        for word_output, word_pointer_num_pos in zip(word_outputs, word_pointer_num_poses):
            num_pointer_embedding = self.get_pointer_embedding(word_pointer_num_pos, word_output)
            num_pointer_embeddings.append(num_pointer_embedding)
        pointer_embedding = torch.cat([embedding.unsqueeze(0) for embedding in num_pointer_embeddings], dim=0).sum(dim=0)
        
        all_embedding = torch.cat((generator_embedding, pointer_embedding), dim=1)
        pointer_embedding_mask = (pointer_embedding, pointer_mask)
        return pointer_embedding_mask, const_embedding, op_embedding, all_embedding

    def fix_duplicate_target(self, nodes_output, targets, duplicate_nums):
        for data_idx, duplicate_num_dict in enumerate(duplicate_nums):
            if len(duplicate_num_dict) > 0:
                k_cls = targets[data_idx].item()
                if k_cls in duplicate_num_dict.keys():
                    node_output = nodes_output[data_idx]
                    candidate_cls = duplicate_num_dict[k_cls]
                    max_cls_pos = node_output.index_select(dim=-1, index=candidate_cls).topk(1)[1][0].item()
                    max_cls = candidate_cls[max_cls_pos].item()
                    targets[data_idx] = max_cls
        return targets

    def init_stacks(self, encoder_hidden):
        batch_size = encoder_hidden.size(0)
        node_stacks = [[NodeEmbeddingNode(hidden, None, None)] for hidden in encoder_hidden]
        tree_stacks = [[] for _ in range(batch_size)]
        return node_stacks, tree_stacks

    def forward_step(self, node_stacks, tree_stacks, nodes_hidden, encoder_outputs, masks, embedding_masks, finished_mask, decoder_nodes_class=None, duplicate_nums=None):
        nodes_output, nodes_context = self.predict(nodes_hidden, encoder_outputs, masks, embedding_masks, finished_mask)
        # embed_index_order => class_order
        nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
        # teacher
        if decoder_nodes_class is not None:
            if duplicate_nums is not None:
                nodes_class = self.fix_duplicate_target(nodes_output, decoder_nodes_class, duplicate_nums)
            else:
                nodes_class = decoder_nodes_class
        # no teacher
        else:
            nodes_class = nodes_output.topk(1)[1].view(-1)
        # class_order => embed_index_order
        embed_nodes_index = self.class_to_embed_index[nodes_class]
        labels_embedding = self.tree_embedding(embedding_masks[-1], tree_stacks, embed_nodes_index)
        nodes_hidden, finished_mask = self.decompose(node_stacks, tree_stacks, nodes_context, labels_embedding)
        return nodes_output, nodes_class, nodes_hidden, finished_mask
    
    def forward_teacher(self, decoder_nodes_label, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length=None, duplicate_nums=None):
        decoder_outputs_list = []
        sequence_symbols_list = []
        batch_size = decoder_init_hidden.size(0)
        decoder_hidden = decoder_init_hidden
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        finished_mask = torch.zeros(batch_size).long()
        if self.use_cuda:
            finished_mask = finished_mask.cuda()
        if decoder_nodes_label is not None:
            seq_len = decoder_nodes_label.size(1)
        else:
            seq_len = max_length
        for di in range(seq_len):
            if decoder_nodes_label is not None:
                decoder_node_class = decoder_nodes_label[:, di]
            else:
                decoder_node_class = None
            decoder_output, symbols, decoder_hidden, finished_mask = self.forward_step(node_stacks, tree_stacks, decoder_hidden, encoder_outputs, masks, embedding_masks, finished_mask=finished_mask, decoder_nodes_class=decoder_node_class,
            duplicate_nums=duplicate_nums)
            decoder_outputs_list.append(decoder_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, sequence_symbols_list

    def forward_beam(self, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width):
        # only support batch_size == 1
        batch_size = decoder_init_hidden.size(0)
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        finished_mask = torch.zeros(batch_size).long()
        if self.use_cuda:
            finished_mask = finished_mask.cuda()
        beams = [BeamNode(0, decoder_init_hidden, finished_mask, node_stacks, tree_stacks, [], [])]
        for _ in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                # finished stack-guided decoding
                if len(b.node_stacks) == 0:
                    current_beams.append(b)
                    continue
                # unfinished decoding
                # batch_size == 1
                # batch_size * class_size
                nodes_output, nodes_context = self.predict(b.nodes_hidden, encoder_outputs, masks, embedding_masks, b.finished_mask)
                # embed_index_order => class_order
                nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
                # batch_size * beam_width
                top_value, top_index = nodes_output.topk(beam_width)
                top_value = torch.exp(top_value)
                for predict_score, predicted_symbol in zip(top_value.split(1, dim=-1), top_index.split(1, dim=-1)):
                    nb = b.copy()
                    # class_order => embed_index_order
                    predicted_symbol = predicted_symbol.view(-1)
                    embed_nodes_index = self.class_to_embed_index[predicted_symbol]
                    labels_embedding = self.tree_embedding(embedding_masks[-1], nb.tree_stacks, embed_nodes_index)
                    nodes_hidden, finished_mask = self.decompose(nb.node_stacks, nb.tree_stacks, nodes_context, labels_embedding, pad_node=False)

                    nb.score = b.score + predict_score.item()
                    nb.nodes_hidden = nodes_hidden
                    nb.finished_mask = finished_mask
                    nb.decoder_outputs_list.append(nodes_output)
                    nb.sequence_symbols_list.append(predicted_symbol)
                    current_beams.append(nb)
            beams = sorted(current_beams, key=lambda b:b.score, reverse=True)
            beams = beams[:beam_width]
            all_finished = True
            for b in beams:
                if len(b.node_stacks[0]) != 0:
                    all_finished = False
                    break
            if all_finished:
                break
        output = beams[0]
        return output.decoder_outputs_list, output.sequence_symbols_list

    def forward(self, encoder_hidden, encoder_outputs, input_lengths, span_length, num_pos, targets=None, max_length=None, beam_width=None, duplicate_nums=None):
        masks = self.get_pad_masks(encoder_outputs, input_lengths, span_length)
        embedding_masks = self.get_class_embedding_mask(num_pos, encoder_outputs)

        decoder_init_hidden = encoder_hidden

        if max_length is None:
            if targets is not None:
                max_length = targets.size(1)
            else:
                max_length = 40
        
        if beam_width is not None:
            return self.forward_beam(decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width)
        else:
            return self.forward_teacher(targets, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, duplicate_nums)
