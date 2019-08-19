
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load(u'ner')
sentence = Sentence(u'George Washington went to Washington .')
tagger.predict(sentence)
print((u'Analysing %s' % sentence))
print(u'\nThe following NER tags are found: \n')
print(sentence.to_tagged_string())
