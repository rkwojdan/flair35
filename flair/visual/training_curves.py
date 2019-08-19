
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import matplotlib.pyplot as plt
from collections import defaultdict
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
from typing import Union
import numpy as np
import csv
import matplotlib
import math
matplotlib.use(u'Agg')
WEIGHT_NAME = 1
WEIGHT_NUMBER = 2
WEIGHT_VALUE = 3


class Plotter(object):
    u"\n    Plots training parameters (loss, f-score, and accuracy) and training weights over time.\n    Input files are the output files 'loss.tsv' and 'weights.txt' from training either a sequence tagger or text\n    classification model.\n    "

    @staticmethod
    def _extract_evaluation_data(file_name):
        training_curves = {
            u'train': {
                u'loss': [],
                u'f_score': [],
                u'acc': [],
            },
            u'test': {
                u'loss': [],
                u'f_score': [],
                u'acc': [],
            },
            u'dev': {
                u'loss': [],
                u'f_score': [],
                u'acc': [],
            },
        }
        with open(file_name, u'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=u'\t')
            row = next(tsvin, None)
            TRAIN_LOSS = row.index(u'TRAIN_LOSS')
            TRAIN_F_SCORE = row.index(u'TRAIN_F-SCORE')
            TRAIN_ACCURACY = row.index(u'TRAIN_ACCURACY')
            DEV_LOSS = row.index(u'DEV_LOSS')
            DEV_F_SCORE = row.index(u'DEV_F-SCORE')
            DEV_ACCURACY = row.index(u'DEV_ACCURACY')
            TEST_LOSS = row.index(u'TEST_LOSS')
            TEST_F_SCORE = row.index(u'TEST_F-SCORE')
            TEST_ACCURACY = row.index(u'TEST_ACCURACY')
            for row in tsvin:
                if (row[TRAIN_LOSS] != u'_'):
                    training_curves[u'train'][u'loss'].append(
                        float(row[TRAIN_LOSS]))
                if (row[TRAIN_F_SCORE] != u'_'):
                    training_curves[u'train'][u'f_score'].append(
                        float(row[TRAIN_F_SCORE]))
                if (row[TRAIN_ACCURACY] != u'_'):
                    training_curves[u'train'][u'acc'].append(
                        float(row[TRAIN_ACCURACY]))
                if (row[DEV_LOSS] != u'_'):
                    training_curves[u'dev'][u'loss'].append(
                        float(row[DEV_LOSS]))
                if (row[DEV_F_SCORE] != u'_'):
                    training_curves[u'dev'][u'f_score'].append(
                        float(row[DEV_F_SCORE]))
                if (row[DEV_ACCURACY] != u'_'):
                    training_curves[u'dev'][u'acc'].append(
                        float(row[DEV_ACCURACY]))
                if (row[TEST_LOSS] != u'_'):
                    training_curves[u'test'][u'loss'].append(
                        float(row[TEST_LOSS]))
                if (row[TEST_F_SCORE] != u'_'):
                    training_curves[u'test'][u'f_score'].append(
                        float(row[TEST_F_SCORE]))
                if (row[TEST_ACCURACY] != u'_'):
                    training_curves[u'test'][u'acc'].append(
                        float(row[TEST_ACCURACY]))
        return training_curves

    @staticmethod
    def _extract_weight_data(file_name):
        weights = defaultdict((lambda: defaultdict((lambda: list()))))
        with open(file_name, u'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=u'\t')
            for row in tsvin:
                name = row[WEIGHT_NAME]
                param = row[WEIGHT_NUMBER]
                value = float(row[WEIGHT_VALUE])
                weights[name][param].append(value)
        return weights

    @staticmethod
    def _extract_learning_rate(file_name):
        lrs = []
        losses = []
        with open(file_name, u'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=u'\t')
            row = next(tsvin, None)
            LEARNING_RATE = row.index(u'LEARNING_RATE')
            TRAIN_LOSS = row.index(u'TRAIN_LOSS')
            for row in tsvin:
                if (row[TRAIN_LOSS] != u'_'):
                    losses.append(float(row[TRAIN_LOSS]))
                if (row[LEARNING_RATE] != u'_'):
                    lrs.append(float(row[LEARNING_RATE]))
        return (lrs, losses)

    def plot_weights(self, file_name):
        if (type(file_name) is unicode):
            file_name = Path(file_name)
        weights = self._extract_weight_data(file_name)
        total = len(weights)
        columns = 2
        rows = max(2, int(math.ceil((total / columns))))
        figsize = (5, 5)
        if (rows != columns):
            figsize = (5, (rows + 5))
        fig = plt.figure()
        (f, axarr) = plt.subplots(rows, columns, figsize=figsize)
        c = 0
        r = 0
        for (name, values) in weights.items():
            axarr[(r, c)].set_title(name, fontsize=6)
            for (_, v) in values.items():
                axarr[(r, c)].plot(np.arange(0, len(v)), v, linewidth=0.35)
            axarr[(r, c)].set_yticks([])
            axarr[(r, c)].set_xticks([])
            c += 1
            if (c == columns):
                c = 0
                r += 1
        while ((r != rows) and (c != columns)):
            axarr[(r, c)].set_yticks([])
            axarr[(r, c)].set_xticks([])
            c += 1
            if (c == columns):
                c = 0
                r += 1
        f.subplots_adjust(hspace=0.5)
        plt.tight_layout(pad=1.0)
        path = (file_name.parent / u'weights.png')
        plt.savefig(path, dpi=300)
        plt.close(fig)

    def plot_training_curves(self, file_name):
        if (type(file_name) is unicode):
            file_name = Path(file_name)
        fig = plt.figure(figsize=(15, 10))
        training_curves = self._extract_evaluation_data(file_name)
        plt.subplot(3, 1, 1)
        if training_curves[u'train'][u'loss']:
            x = np.arange(0, len(training_curves[u'train'][u'loss']))
            plt.plot(x, training_curves[u'train']
                     [u'loss'], label=u'training loss')
        if training_curves[u'dev'][u'loss']:
            x = np.arange(0, len(training_curves[u'dev'][u'loss']))
            plt.plot(x, training_curves[u'dev']
                     [u'loss'], label=u'validation loss')
        if training_curves[u'test'][u'loss']:
            x = np.arange(0, len(training_curves[u'test'][u'loss']))
            plt.plot(x, training_curves[u'test'][u'loss'], label=u'test loss')
        plt.legend(bbox_to_anchor=(1.04, 0),
                   loc=u'lower left', borderaxespad=0)
        plt.ylabel(u'loss')
        plt.xlabel(u'epochs')
        plt.subplot(3, 1, 2)
        if training_curves[u'train'][u'acc']:
            x = np.arange(0, len(training_curves[u'train'][u'acc']))
            plt.plot(x, training_curves[u'train']
                     [u'acc'], label=u'training accuracy')
        if training_curves[u'dev'][u'acc']:
            x = np.arange(0, len(training_curves[u'dev'][u'acc']))
            plt.plot(x, training_curves[u'dev']
                     [u'acc'], label=u'validation accuracy')
        if training_curves[u'test'][u'acc']:
            x = np.arange(0, len(training_curves[u'test'][u'acc']))
            plt.plot(x, training_curves[u'test']
                     [u'acc'], label=u'test accuracy')
        plt.legend(bbox_to_anchor=(1.04, 0),
                   loc=u'lower left', borderaxespad=0)
        plt.ylabel(u'accuracy')
        plt.xlabel(u'epochs')
        plt.subplot(3, 1, 3)
        if training_curves[u'train'][u'f_score']:
            x = np.arange(0, len(training_curves[u'train'][u'f_score']))
            plt.plot(x, training_curves[u'train']
                     [u'f_score'], label=u'training f1-score')
        if training_curves[u'dev'][u'f_score']:
            x = np.arange(0, len(training_curves[u'dev'][u'f_score']))
            plt.plot(x, training_curves[u'dev']
                     [u'f_score'], label=u'validation f1-score')
        if training_curves[u'test'][u'f_score']:
            x = np.arange(0, len(training_curves[u'test'][u'f_score']))
            plt.plot(x, training_curves[u'test']
                     [u'f_score'], label=u'test f1-score')
        plt.legend(bbox_to_anchor=(1.04, 0),
                   loc=u'lower left', borderaxespad=0)
        plt.ylabel(u'f1-score')
        plt.xlabel(u'epochs')
        plt.tight_layout(pad=1.0)
        path = (file_name.parent / u'training.png')
        plt.savefig(path, dpi=300)
        plt.close(fig)

    def plot_learning_rate(self, file_name, skip_first=10, skip_last=5):
        if (type(file_name) is unicode):
            file_name = Path(file_name)
        (lrs, losses) = self._extract_learning_rate(file_name)
        lrs = (lrs[skip_first:(- skip_last)]
               if (skip_last > 0) else lrs[skip_first:])
        losses = (losses[skip_first:(- skip_last)]
                  if (skip_last > 0) else losses[skip_first:])
        (fig, ax) = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel(u'Loss')
        ax.set_xlabel(u'Learning Rate')
        ax.set_xscale(u'log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(u'%.0e'))
        plt.tight_layout(pad=1.0)
        path = (file_name.parent / u'learning_rate.png')
        plt.savefig(path, dpi=300)
        plt.close(fig)
