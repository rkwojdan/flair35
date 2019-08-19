
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch.nn
from abc import abstractmethod
from typing import Union, List
from flair.data import Sentence, Label


class Model(torch.nn.Module):
    u'Abstract base class for all models. Every new type of model must implement these methods.'

    @abstractmethod
    def forward_loss(self, sentences):
        u'Performs a forward pass and returns the loss.'
        pass

    @abstractmethod
    def forward_labels_and_loss(self, sentences):
        u'Predicts the labels/tags for the given list of sentences. Returns the list of labels plus the loss.'
        pass

    @abstractmethod
    def predict(self, sentences, mini_batch_size=32):
        u'Predicts the labels/tags for the given list of sentences. The labels/tags are added directly to the\n        sentences.'
        pass


class LockedDropout(torch.nn.Module):
    u'\n    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.\n    '

    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if ((not self.training) or (not self.dropout_rate)):
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(
            (1 - self.dropout_rate))
        mask = (torch.autograd.Variable(
            m, requires_grad=False) / (1 - self.dropout_rate))
        mask = mask.expand_as(x)
        return (mask * x)


class WordDropout(torch.nn.Module):
    u'\n    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.\n    '

    def __init__(self, dropout_rate=0.05):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if ((not self.training) or (not self.dropout_rate)):
            return x
        m = x.data.new(x.size(0), 1, 1).bernoulli_((1 - self.dropout_rate))
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(x)
        return (mask * x)
