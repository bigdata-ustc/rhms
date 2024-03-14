# -*- coding:utf-8 -*-

from torch import nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        return

    def forward(self, inputs, input_lengths, span_length, trees, words, num_pos, targets=None, max_length=None, beam_width=None, duplicate_nums=None):
        encoder_outputs, encoder_hidden = self.encoder(
            inputs=inputs,
            input_lengths=input_lengths,
            span_length=span_length,
            trees=trees,
            words=words
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
