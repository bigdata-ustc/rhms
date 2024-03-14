# -*- coding:utf-8 -*-

from torch import nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        return

    def forward(self, inputs, input_lengths, span_length, trees, words,
                neighbor_inputs=None, neighbor_input_lengths=None, neighbor_span_length=None, neighbor_trees=None, neighbor_words=None,
                graph_map=None, num_pos=None, targets=None, max_length=None, beam_width=None, duplicate_nums=None, warm_training=False):
        encoder_outputs, encoder_hidden = self.encoder(
            inputs=inputs,
            input_lengths=input_lengths,
            span_length=span_length,
            trees=trees,
            words=words,
            neighbor_inputs=neighbor_inputs,
            neighbor_input_lengths=neighbor_input_lengths,
            neighbor_span_length=neighbor_span_length,
            neighbor_trees=neighbor_trees,
            neighbor_words=neighbor_words,
            graph_map=graph_map,
            warm_training=warm_training
        )

        output = self.decoder(
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            input_lengths=input_lengths,
            span_length=span_length,
            num_pos=num_pos,
            targets=targets,
            max_length=max_length,
            beam_width=beam_width,
            duplicate_nums=duplicate_nums
        )
        return output
