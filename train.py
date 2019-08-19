
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from typing import List
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
corpus = NLPTaskDataFetcher.load_corpus(NLPTask.UD_ENGLISH)
print(corpus)
tag_type = u'pos'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
embedding_types = [WordEmbeddings(u'glove')]
embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger = SequenceTagger(hidden_size=256, embeddings=embeddings,
                        tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=True)
trainer = ModelTrainer(tagger, corpus)
trainer.train(u'resources/taggers/example-ner', EvaluationMetric.MICRO_F1_SCORE,
              learning_rate=0.1, mini_batch_size=32, max_epochs=20, test_mode=True)
plotter = Plotter()
plotter.plot_training_curves(u'resources/taggers/example-ner/loss.tsv')
plotter.plot_weights(u'resources/taggers/example-ner/weights.txt')
