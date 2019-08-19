
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from enum import Enum


class Parameter(Enum):
    EMBEDDINGS = u'embeddings'
    HIDDEN_SIZE = u'hidden_size'
    USE_CRF = u'use_crf'
    USE_RNN = u'use_rnn'
    RNN_LAYERS = u'rnn_layers'
    DROPOUT = u'dropout'
    WORD_DROPOUT = u'word_dropout'
    LOCKED_DROPOUT = u'locked_dropout'
    LEARNING_RATE = u'learning_rate'
    MINI_BATCH_SIZE = u'mini_batch_size'
    ANNEAL_FACTOR = u'anneal_factor'
    ANNEAL_WITH_RESTARTS = u'anneal_with_restarts'
    PATIENCE = u'patience'
    REPROJECT_WORDS = u'reproject_words'
    REPROJECT_WORD_DIMENSION = u'reproject_words_dimension'
    BIDIRECTIONAL = u'bidirectional'
    OPTIMIZER = u'optimizer'
    MOMENTUM = u'momentum'
    DAMPENING = u'dampening'
    WEIGHT_DECAY = u'weight_decay'
    NESTEROV = u'nesterov'
    AMSGRAD = u'amsgrad'
    BETAS = u'betas'
    EPS = u'eps'


TRAINING_PARAMETERS = [Parameter.LEARNING_RATE.value, Parameter.MINI_BATCH_SIZE.value, Parameter.ANNEAL_FACTOR.value, Parameter.PATIENCE.value, Parameter.ANNEAL_WITH_RESTARTS.value,
                       Parameter.MOMENTUM.value, Parameter.DAMPENING.value, Parameter.WEIGHT_DECAY.value, Parameter.NESTEROV.value, Parameter.AMSGRAD.value, Parameter.BETAS.value, Parameter.EPS.value]
SEQUENCE_TAGGER_PARAMETERS = [Parameter.EMBEDDINGS.value, Parameter.HIDDEN_SIZE.value, Parameter.RNN_LAYERS.value,
                              Parameter.USE_CRF.value, Parameter.USE_RNN.value, Parameter.DROPOUT.value, Parameter.LOCKED_DROPOUT.value, Parameter.WORD_DROPOUT.value]
MODEL_TRAINER_PARAMETERS = [Parameter.OPTIMIZER.value]
DOCUMENT_EMBEDDING_PARAMETERS = [Parameter.EMBEDDINGS.value, Parameter.HIDDEN_SIZE.value, Parameter.RNN_LAYERS.value, Parameter.REPROJECT_WORDS.value,
                                 Parameter.REPROJECT_WORD_DIMENSION.value, Parameter.BIDIRECTIONAL.value, Parameter.DROPOUT.value, Parameter.LOCKED_DROPOUT.value, Parameter.WORD_DROPOUT.value]
