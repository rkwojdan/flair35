
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
from abc import abstractmethod
from enum import Enum
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
from typing import Tuple, Union
import numpy as np
from hyperopt import hp, fmin, tpe
import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentLSTMEmbeddings, DocumentPoolEmbeddings
from flair.hyperparameter import Parameter
from flair.hyperparameter.parameter import SEQUENCE_TAGGER_PARAMETERS, TRAINING_PARAMETERS, DOCUMENT_EMBEDDING_PARAMETERS, MODEL_TRAINER_PARAMETERS
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, log_line, init_output_file, add_file_handler
log = logging.getLogger(u'flair')


class OptimizationValue(Enum):
    DEV_LOSS = u'loss'
    DEV_SCORE = u'score'


class SearchSpace(object):

    def __init__(self):
        self.search_space = {

        }

    def add(self, parameter, func, **kwargs):
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice(u'parameters', [self.search_space])


class ParamSelector(object):

    def __init__(self, corpus, base_path, max_epochs, evaluation_metric, training_runs, optimization_value):
        if (type(base_path) is unicode):
            base_path = Path(base_path)
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs
        self.optimization_value = optimization_value
        self.param_selection_file = init_output_file(
            base_path, u'param_selection.txt')

    @abstractmethod
    def _set_up_model(self, params):
        pass

    def _objective(self, params):
        log_line(log)
        log.info(u''.join([u'Evaluation run: ', u'{}'.format(self.run)]))
        log.info(u''.join([u'Evaluating parameter combination:']))
        for (k, v) in params.items():
            if isinstance(v, Tuple):
                v = u','.join([unicode(x) for x in v])
            log.info(u''.join([u'\t', u'{}'.format(k),
                               u': ', u'{}'.format(unicode(v))]))
        log_line(log)
        for sent in self.corpus.get_all_sentences():
            sent.clear_embeddings()
        scores = []
        vars = []
        for i in range(0, self.training_runs):
            log_line(log)
            log.info(u''.join([u'Training run: ', u'{}'.format((i + 1))]))
            model = self._set_up_model(params)
            training_params = {key: params[key]
                               for key in params if (key in TRAINING_PARAMETERS)}
            model_trainer_parameters = {
                key: params[key] for key in params if (key in MODEL_TRAINER_PARAMETERS)}
            trainer = ModelTrainer(model, self.corpus, **
                                   model_trainer_parameters)
            result = trainer.train(self.base_path, evaluation_metric=self.evaluation_metric,
                                   max_epochs=self.max_epochs, param_selection_mode=True, **training_params)
            if (self.optimization_value == OptimizationValue.DEV_LOSS):
                curr_scores = result[u'dev_loss_history'][(- 3):]
            else:
                curr_scores = list(
                    map((lambda s: (1 - s)), result[u'dev_score_history'][(- 3):]))
            score = (sum(curr_scores) / float(len(curr_scores)))
            var = np.var(curr_scores)
            scores.append(score)
            vars.append(var)
        final_score = (sum(scores) / float(len(scores)))
        final_var = (sum(vars) / float(len(vars)))
        test_score = result[u'test_score']
        log_line(log)
        log.info(u''.join([u'Done evaluating parameter combination:']))
        for (k, v) in params.items():
            if isinstance(v, Tuple):
                v = u','.join([unicode(x) for x in v])
            log.info(
                u''.join([u'\t', u'{}'.format(k), u': ', u'{}'.format(v)]))
        log.info(u''.join(
            [u'{}'.format(self.optimization_value.value), u': ', u'{}'.format(final_score)]))
        log.info(u''.join([u'variance: ', u'{}'.format(final_var)]))
        log.info(u''.join([u'test_score: ', u'{}'.format(test_score), u'\n']))
        log_line(log)
        with open(self.param_selection_file, u'a') as f:
            f.write(
                u''.join([u'evaluation run ', u'{}'.format(self.run), u'\n']))
            for (k, v) in params.items():
                if isinstance(v, Tuple):
                    v = u','.join([unicode(x) for x in v])
                f.write(u''.join([u'\t', u'{}'.format(k),
                                  u': ', u'{}'.format(unicode(v)), u'\n']))
            f.write(u''.join([u'{}'.format(
                self.optimization_value.value), u': ', u'{}'.format(final_score), u'\n']))
            f.write(u''.join([u'variance: ', u'{}'.format(final_var), u'\n']))
            f.write(
                u''.join([u'test_score: ', u'{}'.format(test_score), u'\n']))
            f.write(((u'-' * 100) + u'\n'))
        self.run += 1
        return {
            u'status': u'ok',
            u'loss': final_score,
            u'loss_variance': final_var,
        }

    def optimize(self, space, max_evals=100):
        search_space = space.search_space
        best = fmin(self._objective, search_space,
                    algo=tpe.suggest, max_evals=max_evals)
        log_line(log)
        log.info(u'Optimizing parameter configuration done.')
        log.info(u'Best parameter configuration found:')
        for (k, v) in best.items():
            log.info(
                u''.join([u'\t', u'{}'.format(k), u': ', u'{}'.format(v)]))
        log_line(log)
        with open(self.param_selection_file, u'a') as f:
            f.write(u'best parameter combination\n')
            for (k, v) in best.items():
                if isinstance(v, Tuple):
                    v = u','.join([unicode(x) for x in v])
                f.write(u''.join([u'\t', u'{}'.format(k),
                                  u': ', u'{}'.format(unicode(v)), u'\n']))


class SequenceTaggerParamSelector(ParamSelector):

    def __init__(self, corpus, tag_type, base_path, max_epochs=50, evaluation_metric=EvaluationMetric.MICRO_F1_SCORE, training_runs=1, optimization_value=OptimizationValue.DEV_LOSS):
        u'\n        :param corpus: the corpus\n        :param tag_type: tag type to use\n        :param base_path: the path to the result folder (results will be written to that folder)\n        :param max_epochs: number of epochs to perform on every evaluation run\n        :param evaluation_metric: evaluation metric used during training\n        :param training_runs: number of training runs per evaluation run\n        :param optimization_value: value to optimize\n        '
        super(SequenceTaggerParamSelector, self).__init__(corpus, base_path,
                                                          max_epochs, evaluation_metric, training_runs, optimization_value)
        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

    def _set_up_model(self, params):
        sequence_tagger_params = {key: params[key] for key in params if (
            key in SEQUENCE_TAGGER_PARAMETERS)}
        tagger = SequenceTagger(tag_dictionary=self.tag_dictionary,
                                tag_type=self.tag_type, **sequence_tagger_params)
        return tagger


class TextClassifierParamSelector(ParamSelector):

    def __init__(self, corpus, multi_label, base_path, document_embedding_type, max_epochs=50, evaluation_metric=EvaluationMetric.MICRO_F1_SCORE, training_runs=1, optimization_value=OptimizationValue.DEV_LOSS):
        u"\n        :param corpus: the corpus\n        :param multi_label: true, if the dataset is multi label, false otherwise\n        :param base_path: the path to the result folder (results will be written to that folder)\n        :param document_embedding_type: either 'lstm', 'mean', 'min', or 'max'\n        :param max_epochs: number of epochs to perform on every evaluation run\n        :param evaluation_metric: evaluation metric used during training\n        :param training_runs: number of training runs per evaluation run\n        :param optimization_value: value to optimize\n        "
        super(TextClassifierParamSelector, self).__init__(corpus, base_path,
                                                          max_epochs, evaluation_metric, training_runs, optimization_value)
        self.multi_label = multi_label
        self.document_embedding_type = document_embedding_type
        self.label_dictionary = self.corpus.make_label_dictionary()

    def _set_up_model(self, params):
        embdding_params = {key: params[key] for key in params if (
            key in DOCUMENT_EMBEDDING_PARAMETERS)}
        if (self.document_embedding_type == u'lstm'):
            document_embedding = DocumentLSTMEmbeddings(**embdding_params)
        else:
            document_embedding = DocumentPoolEmbeddings(**embdding_params)
        text_classifier = TextClassifier(label_dictionary=self.label_dictionary,
                                         multi_label=self.multi_label, document_embeddings=document_embedding)
        return text_classifier
