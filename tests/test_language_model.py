
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import shutil
from flair.data import Dictionary
from flair.trainers.language_model_trainer import TextCorpus


def test_train_resume_language_model_training(resources_path, results_base_path, tasks_base_path):
    dictionary = Dictionary.load(u'chars')
    corpus = TextCorpus((resources_path / u'corpora/lorem_ipsum'),
                        dictionary, forward=True, character_level=True)
    assert (corpus.test is not None)
    assert (corpus.train is not None)
    assert (corpus.valid is not None)
    assert (len(corpus.train) == 2)


def test_generate_text_with_small_temperatures():
    from flair.embeddings import FlairEmbeddings
    language_model = FlairEmbeddings(u'news-forward-fast').lm
    (text, likelihood) = language_model.generate_text(
        temperature=0.01, number_of_characters=100)
    assert (text is not None)
    assert (len(text) >= 100)
