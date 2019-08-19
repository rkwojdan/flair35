
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import pytest
from flair.visual import *
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
import numpy
from flair.visual.manifold import Visualizer, tSNE
from flair.visual.training_curves import Plotter


def test_highlighter(resources_path):
    with (resources_path / u'visual/snippet.txt').open() as f:
        sentences = [x for x in f.read().split(u'\n') if x]
    embeddings = FlairEmbeddings(u'news-forward')
    features = embeddings.lm.get_representation(sentences[0]).squeeze()
    Highlighter().highlight_selection(features, sentences[0], n=1000, file_=unicode(
        (resources_path / u'visual/highligh.html')))
    (resources_path / u'visual/highligh.html').unlink()


def test_plotting_training_curves_and_weights(resources_path):
    plotter = Plotter()
    plotter.plot_training_curves((resources_path / u'visual/loss.tsv'))
    plotter.plot_weights((resources_path / u'visual/weights.txt'))
    (resources_path / u'visual/weights.png').unlink()
    (resources_path / u'visual/training.png').unlink()
