
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import shutil
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import flair
from flair.data_fetcher import NLPTask, NLPTaskDataFetcher


def test_load_imdb_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(u'imdb', tasks_base_path)
    assert (len(corpus.train) == 5)
    assert (len(corpus.dev) == 5)
    assert (len(corpus.test) == 5)


def test_load_ag_news_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(NLPTask.AG_NEWS, tasks_base_path)
    assert (len(corpus.train) == 10)
    assert (len(corpus.dev) == 10)
    assert (len(corpus.test) == 10)


def test_load_sequence_labeling_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(NLPTask.FASHION, tasks_base_path)
    assert (len(corpus.train) == 6)
    assert (len(corpus.dev) == 1)
    assert (len(corpus.test) == 1)


def test_load_germeval_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(NLPTask.GERMEVAL, tasks_base_path)
    assert (len(corpus.train) == 2)
    assert (len(corpus.dev) == 1)
    assert (len(corpus.test) == 1)


def test_load_ud_english_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.UD_ENGLISH, tasks_base_path)
    assert (len(corpus.train) == 6)
    assert (len(corpus.test) == 4)
    assert (len(corpus.dev) == 2)


def test_load_no_dev_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_column_corpus((tasks_base_path / u'fashion_nodev'), {
        0: u'text',
        2: u'ner',
    })
    assert (len(corpus.train) == 5)
    assert (len(corpus.dev) == 1)
    assert (len(corpus.test) == 1)


def test_load_no_dev_data_explicit(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_column_corpus((tasks_base_path / u'fashion_nodev'), {
        0: u'text',
        2: u'ner',
    }, train_file=u'train.tsv', test_file=u'test.tsv')
    assert (len(corpus.train) == 5)
    assert (len(corpus.dev) == 1)
    assert (len(corpus.test) == 1)


def test_multi_corpus(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpora(
        [NLPTask.FASHION, NLPTask.GERMEVAL], tasks_base_path)
    assert (len(corpus.train) == 8)
    assert (len(corpus.dev) == 2)
    assert (len(corpus.test) == 2)


def test_download_load_data(tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(NLPTask.UD_ENGLISH)
    assert (len(corpus.train) == 12543)
    assert (len(corpus.dev) == 2002)
    assert (len(corpus.test) == 2077)
    shutil.rmtree(
        ((Path(flair.file_utils.CACHE_ROOT) / u'datasets') / u'ud_english'))
