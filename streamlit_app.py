import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from random import sample

import streamlit as st


class CharRNNCell(nn.Module):
    """
    Implement the scheme above as a PyTorch module.
    """

    def __init__(self, num_tokens=len(tokens), embedding_size=36,
                 rnn_num_units=100):
        super(self.__class__, self).__init__()

        self.num_units = rnn_num_units
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(embedding_size + rnn_num_units,
                                    rnn_num_units)
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, x, h_prev):
        """
        This method computes h_next(x, h_prev) and log P(x_next | h_next). 
        We'll call it repeatedly to produce the whole sequence.

        :param x: batch of character ids, containing vector of int64 type
        :param h_prev: previous RNN hidden states, containing matrix 
        [batch, rnn_num_units] of float32 type
        """
        # get vector embedding of x
        x_emb = self.embedding(x)
        h_prev = h_prev

        # compute next hidden state using self.rnn_update
        x_and_h = torch.cat([h_prev, x_emb], dim=-1)

        h_next = self.rnn_update(x_and_h)
        h_next = torch.tanh(h_next)

        assert h_next.size() == h_prev.size()

        # compute logits for the next character probs
        logits = self.rnn_to_logits(h_next)

        return h_next, F.log_softmax(logits, -1)

    def initial_state(self, batch_size):
        """Return RNN state before it processes the first input (h0)."""
        return torch.zeros(batch_size, self.num_units, requires_grad=True)


PATH = 'rnn_model.pt'


def main():
    st.text('Hello!')
    char_rnn = CharRNNCell()
    char_rnn.load_state_dict(torch.load(PATH))


if __name__ == '__main__':
    main()
