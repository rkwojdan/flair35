
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pytest
from typing import Tuple
from flair.data import Dictionary, TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings
from flair.models.text_classification_model import TextClassifier


def init(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS, tasks_base_path)
    label_dict = corpus.make_label_dictionary()
    glove_embedding = WordEmbeddings(u'en-glove')
    document_embeddings = DocumentLSTMEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    return (corpus, label_dict, model)


def test_labels_to_indices(tasks_base_path):
    (corpus, label_dict, model) = init(tasks_base_path)
    result = model._labels_to_indices(corpus.train)
    for i in range(len(corpus.train)):
        expected = label_dict.get_idx_for_item(corpus.train[i].labels[0].value)
        actual = result[i].item()
        assert (expected == actual)


def test_labels_to_one_hot(tasks_base_path):
    (corpus, label_dict, model) = init(tasks_base_path)
    result = model._labels_to_one_hot(corpus.train)
    for i in range(len(corpus.train)):
        expected = label_dict.get_idx_for_item(corpus.train[i].labels[0].value)
        actual = result[i]
        for idx in range(len(label_dict)):
            if (idx == expected):
                assert (actual[idx] == 1)
            else:
                assert (actual[idx] == 0)
