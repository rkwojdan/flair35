
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from flair.data import Dictionary
from flair.training_utils import convert_labels_to_one_hot, Metric


def test_metric_get_classes():
    metric = Metric(u'Test')
    metric.add_fn(u'class-1')
    metric.add_fn(u'class-3')
    metric.add_tn(u'class-1')
    metric.add_tp(u'class-2')
    assert (3 == len(metric.get_classes()))
    assert (u'class-1' in metric.get_classes())
    assert (u'class-2' in metric.get_classes())
    assert (u'class-3' in metric.get_classes())


def test_metric_with_classes():
    metric = Metric(u'Test')
    metric.add_tp(u'class-1')
    metric.add_tn(u'class-1')
    metric.add_tn(u'class-1')
    metric.add_fp(u'class-1')
    metric.add_tp(u'class-2')
    metric.add_tn(u'class-2')
    metric.add_tn(u'class-2')
    metric.add_fp(u'class-2')
    for i in range(0, 10):
        metric.add_tp(u'class-3')
    for i in range(0, 90):
        metric.add_fp(u'class-3')
    metric.add_tp(u'class-4')
    metric.add_tn(u'class-4')
    metric.add_tn(u'class-4')
    metric.add_fp(u'class-4')
    assert (metric.precision(u'class-1') == 0.5)
    assert (metric.precision(u'class-2') == 0.5)
    assert (metric.precision(u'class-3') == 0.1)
    assert (metric.precision(u'class-4') == 0.5)
    assert (metric.recall(u'class-1') == 1)
    assert (metric.recall(u'class-2') == 1)
    assert (metric.recall(u'class-3') == 1)
    assert (metric.recall(u'class-4') == 1)
    assert (metric.accuracy() == metric.micro_avg_accuracy())
    assert (metric.f_score() == metric.micro_avg_f_score())
    assert (metric.f_score(u'class-1') == 0.6667)
    assert (metric.f_score(u'class-2') == 0.6667)
    assert (metric.f_score(u'class-3') == 0.1818)
    assert (metric.f_score(u'class-4') == 0.6667)
    assert (metric.accuracy(u'class-1') == 0.75)
    assert (metric.accuracy(u'class-2') == 0.75)
    assert (metric.accuracy(u'class-3') == 0.1)
    assert (metric.accuracy(u'class-4') == 0.75)
    assert (metric.micro_avg_f_score() == 0.2184)
    assert (metric.macro_avg_f_score() == 0.5714)
    assert (metric.micro_avg_accuracy() == 0.1696)
    assert (metric.macro_avg_accuracy() == 0.5875)
    assert (metric.precision() == 0.1226)
    assert (metric.recall() == 1)


def test_convert_labels_to_one_hot():
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item(u'class-1')
    label_dict.add_item(u'class-2')
    label_dict.add_item(u'class-3')
    one_hot = convert_labels_to_one_hot([[u'class-2']], label_dict)
    assert (one_hot[0][0] == 0)
    assert (one_hot[0][1] == 1)
    assert (one_hot[0][2] == 0)
