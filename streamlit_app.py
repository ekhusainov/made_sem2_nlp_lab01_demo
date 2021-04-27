import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import sample
import pickle

import streamlit as st

PATH = 'rnn_model.pt'
MAX_LENGTH = 528


class CharRNNCell(nn.Module):
    """
    Implement the scheme above as a PyTorch module.
    """

    def __init__(self, num_tokens=36, embedding_size=36,
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


def generate_sample(char_rnn, seed_phrase=' ', max_length=MAX_LENGTH,
                    temperature=1.0):
    """
    The function generates text given a phrase of length of at least SEQ_LENGTH.
    :param seed_phrase: prefix characters, the sequence that the RNN 
    is asked to continue
    :param max_length: maximum output length, including seed_phrase length
    :param temperature: coefficient for sampling; higher temperature produces 
    more chaotic outputs, smaller temperature converges to the single 
    most likely output
    """
    x_sequence = [token_to_idx[token] for token in seed_phrase]
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64)
    hid_state = char_rnn.initial_state(batch_size=1)
    hid_state

    # feed the seed phrase if there is any
    for i in range(len(seed_phrase) - 1):
        hid_state
        hid_state, _ = char_rnn(x_sequence[:, i], hid_state)

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        hid_state, logp_next = char_rnn(x_sequence[:, -1], hid_state)
        temp = F.softmax(logp_next / temperature, dim=-1)
        p_next = temp.data.numpy()[0]
        next_ix = np.random.choice(num_tokens, p=p_next)
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        x_sequence = torch.cat([x_sequence, next_ix], dim=-1)

    # x_sequence = x_sequence.to('cpu')
    return ''.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    # st.text('Hello!')
    char_rnn = CharRNNCell()
    char_rnn.load_state_dict(torch.load(PATH))
    name = 'token_to_idx'
    token_to_idx = load_obj(name)
    poem = generate_sample(char_rnn, seed_phrase='начало', temperature=0.5)
    st.text(poem)


if __name__ == '__main__':
    main()
