
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import torch.nn as nn
import torch
import math
from typing import Union, Tuple
from typing import List
from torch.optim import Optimizer
import flair
from flair.data import Dictionary


class LanguageModel(nn.Module):
    u'Container module with an encoder, a recurrent module, and a decoder.'

    def __init__(self, dictionary, is_forward_lm, hidden_size, nlayers, embedding_size=100, nout=None, dropout=0.1):
        super(LanguageModel, self).__init__()
        self.dictionary = dictionary
        self.is_forward_lm = is_forward_lm
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(len(dictionary), embedding_size)
        if (nlayers == 1):
            self.rnn = nn.LSTM(embedding_size, hidden_size, nlayers)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size,
                               nlayers, dropout=dropout)
        self.hidden = None
        self.nout = nout
        if (nout is not None):
            self.proj = nn.Linear(hidden_size, nout)
            self.initialize(self.proj.weight)
            self.decoder = nn.Linear(nout, len(dictionary))
        else:
            self.proj = None
            self.decoder = nn.Linear(hidden_size, len(dictionary))
        self.init_weights()
        self.to(flair.device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.detach().uniform_((- initrange), initrange)
        self.decoder.bias.detach().fill_(0)
        self.decoder.weight.detach().uniform_((- initrange), initrange)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, input, hidden, ordered_sequence_lengths=None):
        encoded = self.encoder(input)
        emb = self.drop(encoded)
        self.rnn.flatten_parameters()
        (output, hidden) = self.rnn(emb, hidden)
        if (self.proj is not None):
            output = self.proj(output)
        output = self.drop(output)
        decoded = self.decoder(output.view(
            (output.size(0) * output.size(1)), output.size(2)))
        return (decoded.view(output.size(0), output.size(1), decoded.size(1)), output, hidden)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).detach()
        return (weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach(), weight.new(self.nlayers, bsz, self.hidden_size).zero_().clone().detach())

    def get_representation(self, strings, chars_per_chunk=512):
        longest = len(strings[0])
        chunks = []
        splice_begin = 0
        for splice_end in range(chars_per_chunk, longest, chars_per_chunk):
            chunks.append([text[splice_begin:splice_end] for text in strings])
            splice_begin = splice_end
        chunks.append([text[splice_begin:longest] for text in strings])
        hidden = self.init_hidden(len(chunks[0]))
        output_parts = []
        for chunk in chunks:
            sequences_as_char_indices = []
            for string in chunk:
                char_indices = [self.dictionary.get_idx_for_item(
                    char) for char in string]
                sequences_as_char_indices.append(char_indices)
            batch = torch.LongTensor(sequences_as_char_indices).transpose(0, 1)
            batch = batch.to(flair.device)
            (prediction, rnn_output, hidden) = self.forward(batch, hidden)
            rnn_output = rnn_output.detach()
            output_parts.append(rnn_output)
        output = torch.cat(output_parts)
        return output

    def get_output(self, text):
        char_indices = [self.dictionary.get_idx_for_item(
            char) for char in text]
        input_vector = torch.LongTensor([char_indices]).transpose(0, 1)
        hidden = self.init_hidden(1)
        (prediction, rnn_output, hidden) = self.forward(input_vector, hidden)
        return self.repackage_hidden(hidden)

    def repackage_hidden(self, h):
        u'Wraps hidden states in new Variables, to detach them from their history.'
        if (type(h) == torch.Tensor):
            return h.clone().detach()
        else:
            return tuple((self.repackage_hidden(v) for v in h))

    def initialize(self, matrix):
        (in_, out_) = matrix.size()
        stdv = math.sqrt((3.0 / (in_ + out_)))
        matrix.detach().uniform_((- stdv), stdv)

    @classmethod
    def load_language_model(cls, model_file):
        state = torch.load(unicode(model_file), map_location=flair.device)
        model = LanguageModel(state[u'dictionary'], state[u'is_forward_lm'], state[u'hidden_size'],
                              state[u'nlayers'], state[u'embedding_size'], state[u'nout'], state[u'dropout'])
        model.load_state_dict(state[u'state_dict'])
        model.eval()
        model.to(flair.device)
        return model

    @classmethod
    def load_checkpoint(cls, model_file):
        state = torch.load(unicode(model_file), map_location=flair.device)
        epoch = (state[u'epoch'] if (u'epoch' in state) else None)
        split = (state[u'split'] if (u'split' in state) else None)
        loss = (state[u'loss'] if (u'loss' in state) else None)
        optimizer_state_dict = (state[u'optimizer_state_dict'] if (
            u'optimizer_state_dict' in state) else None)
        model = LanguageModel(state[u'dictionary'], state[u'is_forward_lm'], state[u'hidden_size'],
                              state[u'nlayers'], state[u'embedding_size'], state[u'nout'], state[u'dropout'])
        model.load_state_dict(state[u'state_dict'])
        model.eval()
        model.to(flair.device)
        return {
            u'model': model,
            u'epoch': epoch,
            u'split': split,
            u'loss': loss,
            u'optimizer_state_dict': optimizer_state_dict,
        }

    def save_checkpoint(self, file, optimizer, epoch, split, loss):
        model_state = {
            u'state_dict': self.state_dict(),
            u'dictionary': self.dictionary,
            u'is_forward_lm': self.is_forward_lm,
            u'hidden_size': self.hidden_size,
            u'nlayers': self.nlayers,
            u'embedding_size': self.embedding_size,
            u'nout': self.nout,
            u'dropout': self.dropout,
            u'optimizer_state_dict': optimizer.state_dict(),
            u'epoch': epoch,
            u'split': split,
            u'loss': loss,
        }
        torch.save(model_state, unicode(file), pickle_protocol=4)

    def save(self, file):
        model_state = {
            u'state_dict': self.state_dict(),
            u'dictionary': self.dictionary,
            u'is_forward_lm': self.is_forward_lm,
            u'hidden_size': self.hidden_size,
            u'nlayers': self.nlayers,
            u'embedding_size': self.embedding_size,
            u'nout': self.nout,
            u'dropout': self.dropout,
        }
        torch.save(model_state, unicode(file), pickle_protocol=4)

    def generate_text(self, prefix=u'\n', number_of_characters=1000, temperature=1.0, break_on_suffix=None):
        if (prefix == u''):
            prefix = u'\n'
        with torch.no_grad():
            characters = []
            idx2item = self.dictionary.idx2item
            hidden = self.init_hidden(1)
            if (len(prefix) > 1):
                char_tensors = []
                for character in prefix[:(- 1)]:
                    char_tensors.append(torch.tensor(
                        self.dictionary.get_idx_for_item(character)).unsqueeze(0).unsqueeze(0))
                input = torch.cat(char_tensors)
                if torch.cuda.is_available():
                    input = input.cuda()
                (prediction, _, hidden) = self.forward(input, hidden)
            input = torch.tensor(self.dictionary.get_idx_for_item(
                prefix[(- 1)])).unsqueeze(0).unsqueeze(0)
            log_prob = 0.0
            for i in range(number_of_characters):
                if torch.cuda.is_available():
                    input = input.cuda()
                (prediction, _, hidden) = self.forward(input, hidden)
                prediction = prediction.squeeze().detach()
                decoder_output = prediction
                prediction = prediction.div(temperature)
                max = torch.max(prediction)
                prediction -= max
                word_weights = prediction.exp().cpu()
                try:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                except:
                    word_idx = torch.tensor(0)
                prob = decoder_output[word_idx]
                log_prob += prob
                input = word_idx.detach().unsqueeze(0).unsqueeze(0)
                word = idx2item[word_idx].decode(u'UTF-8')
                characters.append(word)
                if (break_on_suffix is not None):
                    if u''.join(characters).endswith(break_on_suffix):
                        break
            text = (prefix + u''.join(characters))
            log_prob = log_prob.item()
            log_prob /= len(characters)
            if (not self.is_forward_lm):
                text = text[::(- 1)]
            return (text, log_prob)
