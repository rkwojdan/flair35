
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings
import logging
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
from typing import List, Union
import torch
import torch.nn as nn
import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, Label
from flair.file_utils import cached_path
from flair.training_utils import convert_labels_to_one_hot, clear_embeddings
log = logging.getLogger(u'flair')


class TextClassifier(flair.nn.Model):
    u'\n    Text Classification Model\n    The model takes word embeddings, puts them into an LSTM to obtain a text representation, and puts the\n    text representation in the end into a linear layer to get the actual class label.\n    The model can handle single and multi class data sets.\n    '

    def __init__(self, document_embeddings, label_dictionary, multi_label):
        super(TextClassifier, self).__init__()
        self.document_embeddings = document_embeddings
        self.label_dictionary = label_dictionary
        self.multi_label = multi_label
        self.document_embeddings = document_embeddings
        self.decoder = nn.Linear(
            self.document_embeddings.embedding_length, len(self.label_dictionary))
        self._init_weights()
        if multi_label:
            self.loss_function = nn.BCELoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()
        self.to(flair.device)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences):
        self.document_embeddings.embed(sentences)
        text_embedding_list = [sentence.get_embedding().unsqueeze(0)
                               for sentence in sentences]
        text_embedding_tensor = torch.cat(
            text_embedding_list, 0).to(flair.device)
        label_scores = self.decoder(text_embedding_tensor)
        return label_scores

    def save(self, model_file):
        u'\n        Saves the current model to the provided file.\n        :param model_file: the model file\n        '
        model_state = {
            u'state_dict': self.state_dict(),
            u'document_embeddings': self.document_embeddings,
            u'label_dictionary': self.label_dictionary,
            u'multi_label': self.multi_label,
        }
        torch.save(model_state, unicode(model_file), pickle_protocol=4)

    def save_checkpoint(self, model_file, optimizer_state, scheduler_state, epoch, loss):
        u'\n        Saves the current model to the provided file.\n        :param model_file: the model file\n        '
        model_state = {
            u'state_dict': self.state_dict(),
            u'document_embeddings': self.document_embeddings,
            u'label_dictionary': self.label_dictionary,
            u'multi_label': self.multi_label,
            u'optimizer_state_dict': optimizer_state,
            u'scheduler_state_dict': scheduler_state,
            u'epoch': epoch,
            u'loss': loss,
        }
        torch.save(model_state, unicode(model_file), pickle_protocol=4)

    @classmethod
    def load_from_file(cls, model_file):
        u'\n        Loads the model from the given file.\n        :param model_file: the model file\n        :return: the loaded text classifier model\n        '
        state = TextClassifier._load_state(model_file)
        model = TextClassifier(document_embeddings=state[u'document_embeddings'],
                               label_dictionary=state[u'label_dictionary'], multi_label=state[u'multi_label'])
        model.load_state_dict(state[u'state_dict'])
        model.eval()
        model.to(flair.device)
        return model

    @classmethod
    def load_checkpoint(cls, model_file):
        state = TextClassifier._load_state(model_file)
        model = TextClassifier.load_from_file(model_file)
        epoch = (state[u'epoch'] if (u'epoch' in state) else None)
        loss = (state[u'loss'] if (u'loss' in state) else None)
        optimizer_state_dict = (state[u'optimizer_state_dict'] if (
            u'optimizer_state_dict' in state) else None)
        scheduler_state_dict = (state[u'scheduler_state_dict'] if (
            u'scheduler_state_dict' in state) else None)
        return {
            u'model': model,
            u'epoch': epoch,
            u'loss': loss,
            u'optimizer_state_dict': optimizer_state_dict,
            u'scheduler_state_dict': scheduler_state_dict,
        }

    @classmethod
    def _load_state(cls, model_file):
        with warnings.catch_warnings():
            warnings.filterwarnings(u'ignore')
            f = flair.file_utils.load_big_file(unicode(model_file))
            state = torch.load(f, map_location=flair.device)
            return state

    def forward_loss(self, sentences):
        scores = self.forward(sentences)
        return self._calculate_loss(scores, sentences)

    def forward_labels_and_loss(self, sentences):
        scores = self.forward(sentences)
        labels = self._obtain_labels(scores)
        loss = self._calculate_loss(scores, sentences)
        return (labels, loss)

    def predict(self, sentences, mini_batch_size=32):
        u'\n        Predicts the class labels for the given sentences. The labels are directly added to the sentences.\n        :param sentences: list of sentences\n        :param mini_batch_size: mini batch size to use\n        :return: the list of sentences containing the labels\n        '
        with torch.no_grad():
            if (type(sentences) is Sentence):
                sentences = [sentences]
            filtered_sentences = self._filter_empty_sentences(sentences)
            batches = [filtered_sentences[x:(
                x + mini_batch_size)] for x in range(0, len(filtered_sentences), mini_batch_size)]
            for batch in batches:
                scores = self.forward(batch)
                predicted_labels = self._obtain_labels(scores)
                for (sentence, labels) in zip(batch, predicted_labels):
                    sentence.labels = labels
                clear_embeddings(batch)
            return sentences

    @staticmethod
    def _filter_empty_sentences(sentences):
        filtered_sentences = [
            sentence for sentence in sentences if sentence.tokens]
        if (len(sentences) != len(filtered_sentences)):
            log.warning(u'Ignore {} sentence(s) with no tokens.'.format(
                (len(sentences) - len(filtered_sentences))))
        return filtered_sentences

    def _calculate_loss(self, scores, sentences):
        u'\n        Calculates the loss.\n        :param scores: the prediction scores from the model\n        :param sentences: list of sentences\n        :return: loss value\n        '
        if self.multi_label:
            return self._calculate_multi_label_loss(scores, sentences)
        return self._calculate_single_label_loss(scores, sentences)

    def _obtain_labels(self, scores):
        u'\n        Predicts the labels of sentences.\n        :param scores: the prediction scores from the model\n        :return: list of predicted labels\n        '
        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]
        return [self._get_single_label(s) for s in scores]

    def _get_multi_label(self, label_scores):
        labels = []
        sigmoid = torch.nn.Sigmoid()
        results = list(map((lambda x: sigmoid(x)), label_scores))
        for (idx, conf) in enumerate(results):
            if (conf > 0.5):
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))
        return labels

    def _get_single_label(self, label_scores):
        (conf, idx) = torch.max(label_scores, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())
        return [Label(label, conf.item())]

    def _calculate_multi_label_loss(self, label_scores, sentences):
        sigmoid = nn.Sigmoid()
        return self.loss_function(sigmoid(label_scores), self._labels_to_one_hot(sentences))

    def _calculate_single_label_loss(self, label_scores, sentences):
        return self.loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences):
        label_list = [sentence.get_label_names() for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences):
        indices = [torch.LongTensor([self.label_dictionary.get_idx_for_item(
            label.value) for label in sentence.labels]) for sentence in sentences]
        vec = torch.cat(indices, 0).to(flair.device)
        return vec

    @staticmethod
    def load(model):
        model_file = None
        aws_resource_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4'
        cache_dir = Path(u'models')
        if (model.lower() == u'de-offensive-language'):
            base_path = u'/'.join([aws_resource_path,
                                   u'TEXT-CLASSIFICATION_germ-eval-2018_task-1', u'germ-eval-2018-task-1.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'en-sentiment'):
            base_path = u'/'.join([aws_resource_path,
                                   u'TEXT-CLASSIFICATION_imdb', u'imdb.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        if (model_file is not None):
            return TextClassifier.load_from_file(model_file)
