"""MADE. Второй семестр. NLP. LAB01."""
import pickle
import re

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = 'rnn_model.pt'
PATH_LSTM = 'lstm_3_model.pt'
MAX_LENGTH = 528
NUM_TOKENS = 36
DEVICE = 'cpu'
TOKEN_TO_ID_PATH = 'token_to_idx'


class CharRNNCell(nn.Module):
    """
    Implement the scheme above as a PyTorch module.
    """

    def __init__(self, num_tokens=36, embedding_size=36,
                 rnn_num_units=100):
        super(CharRNNCell, self).__init__()
        self.num_units = rnn_num_units
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.rnn_update = nn.Linear(embedding_size + rnn_num_units,
                                    rnn_num_units)
        self.rnn_to_logits = nn.Linear(rnn_num_units, num_tokens)

    def forward(self, current_x, h_prev):
        """
        This method computes h_next(x, h_prev) and log P(x_next | h_next).
        We'll call it repeatedly to produce the whole sequence.

        :param x: batch of character ids, containing vector of int64 type
        :param h_prev: previous RNN hidden states, containing matrix
        [batch, rnn_num_units] of float32 type
        """
        # get vector embedding of x
        x_emb = self.embedding(current_x)

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


def generate_sample(model, token_to_idx, seed_phrase=' ', max_length=MAX_LENGTH,
                    temperature=1.0, ):
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
    hid_state = model.initial_state(batch_size=1)
    # hid_state

    # feed the seed phrase if there is any
    for i in range(len(seed_phrase) - 1):
        # hid_state
        hid_state, _ = model(x_sequence[:, i], hid_state)

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        hid_state, logp_next = model(x_sequence[:, -1], hid_state)
        temp = F.softmax(logp_next / temperature, dim=-1)
        p_next = temp.data.numpy()[0]
        next_ix = np.random.choice(36, p=p_next)
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        x_sequence = torch.cat([x_sequence, next_ix], dim=-1)

    tokens = list(token_to_idx.keys())
    # x_sequence = x_sequence.to('cpu')
    return ''.join([tokens[ix] for ix in x_sequence.data.numpy()[0]])


class LSTMnet(nn.Module):
    def __init__(self, n_vocab=NUM_TOKENS,
                 seq_size=MAX_LENGTH, embedding_size=36,
                 lstm_size=100):
        super(LSTMnet, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, current_x, prev_state):
        embed = self.embedding(current_x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


def predict_3(net, token_to_idx, idx_to_token, device=DEVICE,
              seed_phrase=' ', max_length=MAX_LENGTH, temperature=1.0):
    net.eval()
    char_list = list(seed_phrase)
    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for current_char in char_list:
        elem_ix = torch.tensor([[token_to_idx[current_char]]]).to(device)
        output, (state_h, state_c) = net(elem_ix, (state_h, state_c))

    p_next = F.softmax(output / temperature, dim=-1).data.numpy()[0][0]

    next_ix = np.random.choice(NUM_TOKENS, p=p_next)
    char_list.append(idx_to_token[next_ix])

    for _ in range(max_length):
        elem_ix = torch.tensor([[next_ix]]).to(device)
        output, (state_h, state_c) = net(elem_ix, (state_h, state_c))

        p_next = F.softmax(output / temperature, dim=-1).data.numpy()[0][0]

        next_ix = np.random.choice(NUM_TOKENS, p=p_next)
        char_list.append(idx_to_token[next_ix])

    return ''.join(char_list)[:-4]


def load_obj(name):
    with open(name + '.pkl', 'rb') as our_file:
        return pickle.load(our_file)


def good_chars(text):
    temp = re.sub(r'[а-я]|ё|\n|-| ', '', text)
    if temp == '':
        return True
    return False


def main():
    # st.text('Hello!')
    token_to_idx = load_obj(TOKEN_TO_ID_PATH)
    idx_to_token = {val: key for key, val in token_to_idx.items()}
    begin = st.text_input(
        'Начало стиха',
    )
    if len(begin) != 0 and good_chars(begin):

        char_rnn = CharRNNCell()
        char_rnn.load_state_dict(torch.load(PATH))
        lstm_3 = LSTMnet()
        lstm_3.load_state_dict(torch.load(PATH_LSTM))

        value_temperature = st.slider(
            label='temperature',
            min_value=0.01,
            max_value=5.0,
            step=0.01,
        )
        poem = generate_sample(
            model=char_rnn,
            seed_phrase=begin,
            temperature=value_temperature,
            token_to_idx=token_to_idx,
        )
        st.markdown("<h3 style='text-align: center;'>RNN</h3>",
                    unsafe_allow_html=True)
        st.text(poem)
        poem_lstm = predict_3(
            net=lstm_3,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            seed_phrase=begin,
            temperature=value_temperature,
        )
        st.markdown("<h3 style='text-align: center;'>LSTM</h3>",
                    unsafe_allow_html=True)
        st.text(poem_lstm)
    elif len(begin) == 0:
        st.text('Введите что-нибудь')
    else:
        st.text('Вводите только строчные русские буквы, пробелы и тире.')


if __name__ == '__main__':
    main()
