
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from abc import abstractmethod
from typing import List, Dict, Union
import torch
import logging
from collections import Counter
from collections import defaultdict
from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer
log = logging.getLogger(u'flair')


class Dictionary(object):
    u'\n    This class holds a dictionary that maps strings to IDs, used to generate one-hot encodings of strings.\n    '

    def __init__(self, add_unk=True):
        self.item2idx = {

        }
        self.idx2item = []
        if add_unk:
            self.add_item(u'<unk>')

    def add_item(self, item):
        u'\n        add string - if already in dictionary returns its ID. if not in dictionary, it will get a new ID.\n        :param item: a string for which to assign an id\n        :return: ID of string\n        '
        item = item.encode(u'utf-8')
        if (item not in self.item2idx):
            self.idx2item.append(item)
            self.item2idx[item] = (len(self.idx2item) - 1)
        return self.item2idx[item]

    def get_idx_for_item(self, item):
        u'\n        returns the ID of the string, otherwise 0\n        :param item: string for which ID is requested\n        :return: ID of string, otherwise 0\n        '
        item = item.encode(u'utf-8')
        if (item in self.item2idx.keys()):
            return self.item2idx[item]
        else:
            return 0

    def get_items(self):
        items = []
        for item in self.idx2item:
            items.append(item.decode(u'UTF-8'))
        return items

    def __len__(self):
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode(u'UTF-8')

    def save(self, savefile):
        try:
            import pickle
        except ImportError:
            import six.moves.cPickle as pickle
        with open(savefile, u'wb') as f:
            mappings = {
                u'idx2item': self.idx2item,
                u'item2idx': self.item2idx,
            }
            pickle.dump(mappings, f)

    @classmethod
    def load_from_file(cls, filename):
        try:
            import pickle
        except ImportError:
            import six.moves.cPickle as pickle
        dictionary = Dictionary()
        with open(filename, u'rb') as f:
            mappings = pickle.load(f, encoding=u'latin1')
            idx2item = mappings[u'idx2item']
            item2idx = mappings[u'item2idx']
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary

    @classmethod
    def load(cls, name):
        from flair.file_utils import cached_path
        if ((name == u'chars') or (name == u'common-chars')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models/common_characters'
            char_dict = cached_path(base_path, cache_dir=u'datasets')
            return Dictionary.load_from_file(char_dict)
        return Dictionary.load_from_file(name)


class Label(object):
    u'\n    This class represents a label of a sentence. Each label has a value and optionally a confidence score. The\n    score needs to be between 0.0 and 1.0. Default value for the score is 1.0.\n    '

    def __init__(self, value, score=1.0):
        self.value = value
        self.score = score
        super(Label, self).__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if ((not value) and (value != u'')):
            raise ValueError(
                u'Incorrect label value provided. Label value needs to be set.')
        else:
            self._value = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if (0.0 <= score <= 1.0):
            self._score = score
        else:
            self._score = 1.0

    def to_dict(self):
        return {
            u'value': self.value,
            u'confidence': self.score,
        }

    def __str__(self):
        return u'{} ({})'.format(self._value, self._score)

    def __repr__(self):
        return u'{} ({})'.format(self._value, self._score)


class Token(object):
    u'\n    This class represents one word in a tokenized sentence. Each token may have any number of tags. It may also point\n    to its head in a dependency tree.\n    '

    def __init__(self, text, idx=None, head_id=None, whitespace_after=True, start_position=None):
        self.text = text
        self.idx = idx
        self.head_id = head_id
        self.whitespace_after = whitespace_after
        self.start_pos = start_position
        self.end_pos = ((start_position + len(text))
                        if (start_position is not None) else None)
        self.sentence = None
        self._embeddings = {

        }
        self.tags = {

        }

    def add_tag_label(self, tag_type, tag):
        self.tags[tag_type] = tag

    def add_tag(self, tag_type, tag_value, confidence=1.0):
        tag = Label(tag_value, confidence)
        self.tags[tag_type] = tag

    def get_tag(self, tag_type):
        if (tag_type in self.tags):
            return self.tags[tag_type]
        return Label(u'')

    def get_head(self):
        return self.sentence.get_token(self.head_id)

    def set_embedding(self, name, vector):
        self._embeddings[name] = vector.cpu()

    def clear_embeddings(self):
        self._embeddings = {

        }

    def get_embedding(self):
        embeddings = [self._embeddings[embed]
                      for embed in sorted(self._embeddings.keys())]
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.Tensor()

    @property
    def start_position(self):
        return self.start_pos

    @property
    def end_position(self):
        return self.end_pos

    @property
    def embedding(self):
        return self.get_embedding()

    def __str__(self):
        return (u'Token: {} {}'.format(self.idx, self.text) if (self.idx is not None) else u'Token: {}'.format(self.text))

    def __repr__(self):
        return (u'Token: {} {}'.format(self.idx, self.text) if (self.idx is not None) else u'Token: {}'.format(self.text))


class Span(object):
    u'\n    This class represents one textual span consisting of Tokens. A span may have a tag.\n    '

    def __init__(self, tokens, tag=None, score=1.0):
        self.tokens = tokens
        self.tag = tag
        self.score = score
        self.start_pos = None
        self.end_pos = None
        if tokens:
            self.start_pos = tokens[0].start_position
            self.end_pos = tokens[(len(tokens) - 1)].end_position

    @property
    def text(self):
        return u' '.join([t.text for t in self.tokens])

    def to_original_text(self):
        unicode = u''
        pos = self.tokens[0].start_pos
        for t in self.tokens:
            while (t.start_pos != pos):
                unicode += u' '
                pos += 1
            unicode += t.text
            pos += len(t.text)
        return unicode

    def to_dict(self):
        return {
            u'text': self.to_original_text(),
            u'start_pos': self.start_pos,
            u'end_pos': self.end_pos,
            u'type': self.tag,
            u'confidence': self.score,
        }

    def __str__(self):
        ids = u','.join([unicode(t.idx) for t in self.tokens])
        return (u'{}-span [{}]: "{}"'.format(self.tag, ids, self.text) if (self.tag is not None) else u'span [{}]: "{}"'.format(ids, self.text))

    def __repr__(self):
        ids = u','.join([unicode(t.idx) for t in self.tokens])
        return (u'<{}-span ({}): "{}">'.format(self.tag, ids, self.text) if (self.tag is not None) else u'<span ({}): "{}">'.format(ids, self.text))


class Sentence(object):
    u'\n    A Sentence is a list of Tokens and is used to represent a sentence or text fragment.\n    '

    def __init__(self, text=None, use_tokenizer=False, labels=None):
        super(Sentence, self).__init__()
        self.tokens = []
        self.labels = []
        if (labels is not None):
            self.add_labels(labels)
        self._embeddings = {

        }
        if (text is not None):
            if use_tokenizer:
                tokens = []
                sentences = split_single(text)
                for sentence in sentences:
                    contractions = split_contractions(word_tokenizer(sentence))
                    tokens.extend(contractions)
                index = text.index
                running_offset = 0
                last_word_offset = (- 1)
                last_token = None
                for word in tokens:
                    try:
                        word_offset = index(word, running_offset)
                        start_position = word_offset
                    except:
                        word_offset = (last_word_offset + 1)
                        start_position = (
                            (running_offset + 1) if (running_offset > 0) else running_offset)
                    token = Token(word, start_position=start_position)
                    self.add_token(token)
                    if (((word_offset - 1) == last_word_offset) and (last_token is not None)):
                        last_token.whitespace_after = False
                    word_len = len(word)
                    running_offset = (word_offset + word_len)
                    last_word_offset = (running_offset - 1)
                    last_token = token
            else:
                word = u''
                for (index, char) in enumerate(text):
                    if (char == u' '):
                        if (len(word) > 0):
                            token = Token(
                                word, start_position=(index - len(word)))
                            self.add_token(token)
                        word = u''
                    else:
                        word += char
                index += 1
                if (len(word) > 0):
                    token = Token(word, start_position=(index - len(word)))
                    self.add_token(token)

    def get_token(self, token_id):
        for token in self.tokens:
            if (token.idx == token_id):
                return token

    def add_token(self, token):
        self.tokens.append(token)
        token.sentence = self
        if (token.idx is None):
            token.idx = len(self.tokens)

    def get_spans(self, tag_type, min_score=(- 1)):
        spans = []
        current_span = []
        tags = defaultdict((lambda: 0.0))
        previous_tag_value = u'O'
        for token in self:
            tag = token.get_tag(tag_type)
            tag_value = tag.value
            if ((tag_value == u'') or (tag_value == u'O')):
                tag_value = u'O-'
            if (tag_value[0:2] not in [u'B-', u'I-', u'O-', u'E-', u'S-']):
                tag_value = (u'S-' + tag_value)
            in_span = False
            if (tag_value[0:2] not in [u'O-']):
                in_span = True
            starts_new_span = False
            if (tag_value[0:2] in [u'B-', u'S-']):
                starts_new_span = True
            if ((previous_tag_value[0:2] in [u'S-']) and (previous_tag_value[2:] != tag_value[2:]) and in_span):
                starts_new_span = True
            if ((starts_new_span or (not in_span)) and (len(current_span) > 0)):
                scores = [t.get_tag(tag_type).score for t in current_span]
                span_score = (sum(scores) / len(scores))
                if (span_score > min_score):
                    spans.append(Span(current_span, tag=sorted(tags.items(), key=(
                        lambda k_v: k_v[1]), reverse=True)[0][0], score=span_score))
                current_span = []
                tags = defaultdict((lambda: 0.0))
            if in_span:
                current_span.append(token)
                weight = (1.1 if starts_new_span else 1.0)
                tags[tag_value[2:]] += weight
            previous_tag_value = tag_value
        if (len(current_span) > 0):
            scores = [t.get_tag(tag_type).score for t in current_span]
            span_score = (sum(scores) / len(scores))
            if (span_score > min_score):
                spans.append(Span(current_span, tag=sorted(tags.items(), key=(
                    lambda k_v: k_v[1]), reverse=True)[0][0], score=span_score))
        return spans

    def add_label(self, label):
        if (type(label) is Label):
            self.labels.append(label)
        elif (type(label) is unicode):
            self.labels.append(Label(label))

    def add_labels(self, labels):
        for label in labels:
            self.add_label(label)

    def get_label_names(self):
        return [label.value for label in self.labels]

    @property
    def embedding(self):
        return self.get_embedding()

    def set_embedding(self, name, vector):
        self._embeddings[name] = vector.cpu()

    def get_embedding(self):
        embeddings = []
        for embed in sorted(self._embeddings.keys()):
            embedding = self._embeddings[embed]
            embeddings.append(embedding)
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.Tensor()

    def clear_embeddings(self, also_clear_word_embeddings=True):
        self._embeddings = {

        }
        if also_clear_word_embeddings:
            for token in self:
                token.clear_embeddings()

    def cpu_embeddings(self):
        for (name, vector) in self._embeddings.items():
            self._embeddings[name] = vector.cpu()

    def to_tagged_string(self, main_tag=None):
        list = []
        for token in self.tokens:
            list.append(token.text)
            tags = []
            for tag_type in token.tags.keys():
                if ((main_tag is not None) and (main_tag != tag_type)):
                    continue
                if ((token.get_tag(tag_type).value == u'') or (token.get_tag(tag_type).value == u'O')):
                    continue
                tags.append(token.get_tag(tag_type).value)
            all_tags = ((u'<' + u'/'.join(tags)) + u'>')
            if (all_tags != u'<>'):
                list.append(all_tags)
        return u' '.join(list)

    def to_tokenized_string(self):
        return u' '.join([t.text for t in self.tokens])

    def to_plain_string(self):
        plain = u''
        for token in self.tokens:
            plain += token.text
            if token.whitespace_after:
                plain += u' '
        return plain.rstrip()

    def convert_tag_scheme(self, tag_type=u'ner', target_scheme=u'iob'):
        tags = []
        for token in self.tokens:
            token = token
            tags.append(token.get_tag(tag_type))
        if (target_scheme == u'iob'):
            iob2(tags)
        if (target_scheme == u'iobes'):
            iob2(tags)
            tags = iob_iobes(tags)
        for (index, tag) in enumerate(tags):
            self.tokens[index].add_tag(tag_type, tag)

    def infer_space_after(self):
        u'\n        Heuristics in case you wish to infer whitespace_after values for tokenized text. This is useful for some old NLP\n        tasks (such as CoNLL-03 and CoNLL-2000) that provide only tokenized data with no info of original whitespacing.\n        :return:\n        '
        last_token = None
        quote_count = 0
        for token in self.tokens:
            if (token.text == u'"'):
                quote_count += 1
                if ((quote_count % 2) != 0):
                    token.whitespace_after = False
                elif (last_token is not None):
                    last_token.whitespace_after = False
            if (last_token is not None):
                if (token.text in [u'.', u':', u',', u';', u')', u"n't", u'!', u'?']):
                    last_token.whitespace_after = False
                if token.text.startswith(u"'"):
                    last_token.whitespace_after = False
            if (token.text in [u'(']):
                token.whitespace_after = False
            last_token = token
        return self

    def to_original_text(self):
        unicode = u''
        pos = 0
        for t in self.tokens:
            while (t.start_pos != pos):
                unicode += u' '
                pos += 1
            unicode += t.text
            pos += len(t.text)
        return unicode

    def to_dict(self, tag_type=None):
        labels = []
        entities = []
        if tag_type:
            entities = [span.to_dict() for span in self.get_spans(tag_type)]
        if self.labels:
            labels = [l.to_dict() for l in self.labels]
        return {
            u'text': self.to_original_text(),
            u'labels': labels,
            u'entities': entities,
        }

    def __getitem__(self, idx):
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return u'Sentence: "{}" - {} Tokens'.format(u' '.join([t.text for t in self.tokens]), len(self))

    def __copy__(self):
        s = Sentence()
        for token in self.tokens:
            nt = Token(token.text)
            for tag_type in token.tags:
                nt.add_tag(tag_type, token.get_tag(tag_type).value,
                           token.get_tag(tag_type).score)
            s.add_token(nt)
        return s

    def __str__(self):
        if self.labels:
            return u''.join([u'Sentence: "', u'{}'.format(self.to_tokenized_string()), u'" - ', u'{}'.format(len(self)), u' Tokens - Labels: ', u'{}'.format(self.labels), u' '])
        else:
            return u''.join([u'Sentence: "', u'{}'.format(self.to_tokenized_string()), u'" - ', u'{}'.format(len(self)), u' Tokens'])

    def __len__(self):
        return len(self.tokens)


class Corpus(object):

    @property
    @abstractmethod
    def train(self):
        pass

    @property
    @abstractmethod
    def dev(self):
        pass

    @property
    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def downsample(self, percentage=0.1, only_downsample_train=False):
        u'Downsamples this corpus to a percentage of the sentences.'
        pass

    @abstractmethod
    def get_all_sentences(self):
        u'Gets all sentences in the corpus (train, dev and test splits together).'
        pass

    @abstractmethod
    def make_tag_dictionary(self, tag_type):
        u'Produces a dictionary of token tags of tag_type.'
        pass

    @abstractmethod
    def make_label_dictionary(self):
        u'\n        Creates a dictionary of all labels assigned to the sentences in the corpus.\n        :return: dictionary of labels\n        '
        pass


class TaggedCorpus(Corpus):

    def __init__(self, train, dev, test, name=u'corpus'):
        self._train = train
        self._dev = dev
        self._test = test
        self.name = name

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test

    def downsample(self, percentage=0.1, only_downsample_train=False):
        self._train = self._downsample_to_proportion(self.train, percentage)
        if (not only_downsample_train):
            self._dev = self._downsample_to_proportion(self.dev, percentage)
            self._test = self._downsample_to_proportion(self.test, percentage)
        return self

    def get_all_sentences(self):
        all_sentences = []
        all_sentences.extend(self.train)
        all_sentences.extend(self.dev)
        all_sentences.extend(self.test)
        return all_sentences

    def make_tag_dictionary(self, tag_type):
        tag_dictionary = Dictionary()
        tag_dictionary.add_item(u'O')
        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                token = token
                tag_dictionary.add_item(token.get_tag(tag_type).value)
        tag_dictionary.add_item(u'<START>')
        tag_dictionary.add_item(u'<STOP>')
        return tag_dictionary

    def make_label_dictionary(self):
        u'\n        Creates a dictionary of all labels assigned to the sentences in the corpus.\n        :return: dictionary of labels\n        '
        labels = set(self._get_all_label_names())
        label_dictionary = Dictionary(add_unk=False)
        for label in labels:
            label_dictionary.add_item(label)
        return label_dictionary

    def make_vocab_dictionary(self, max_tokens=(- 1), min_freq=1):
        u'\n        Creates a dictionary of all tokens contained in the corpus.\n        By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.\n        If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.\n        If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered\n        to be added to the dictionary.\n        :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)\n        :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)\n        :return: dictionary of tokens\n        '
        tokens = self._get_most_common_tokens(max_tokens, min_freq)
        vocab_dictionary = Dictionary()
        for token in tokens:
            vocab_dictionary.add_item(token)
        return vocab_dictionary

    def _get_most_common_tokens(self, max_tokens, min_freq):
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()
        tokens = []
        for (token, freq) in tokens_and_frequencies:
            if (((min_freq != (- 1)) and (freq < min_freq)) or ((max_tokens != (- 1)) and (len(tokens) == max_tokens))):
                break
            tokens.append(token)
        return tokens

    def _get_all_label_names(self):
        return [label.value for sent in self.train for label in sent.labels]

    def _get_all_tokens(self):
        tokens = list(map((lambda s: s.tokens), self.train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    def _downsample_to_proportion(self, list, proportion):
        counter = 0.0
        last_counter = None
        downsampled = []
        for item in list:
            counter += proportion
            if (int(counter) != last_counter):
                downsampled.append(item)
                last_counter = int(counter)
        return downsampled

    def obtain_statistics(self, tag_type=None, pretty_print=True):
        u'\n        Print statistics about the class distribution (only labels of sentences are taken into account) and sentence\n        sizes.\n        '
        json_string = {
            u'TRAIN': self._obtain_statistics_for(self.train, u'TRAIN', tag_type),
            u'TEST': self._obtain_statistics_for(self.test, u'TEST', tag_type),
            u'DEV': self._obtain_statistics_for(self.dev, u'DEV', tag_type),
        }
        if pretty_print:
            import json
            json_string = json.dumps(json_string, indent=4)
        return json_string

    @staticmethod
    def _obtain_statistics_for(sentences, name, tag_type):
        if (len(sentences) == 0):
            return {

            }
        classes_to_count = TaggedCorpus._get_class_to_count(sentences)
        tags_to_count = TaggedCorpus._get_tag_to_count(sentences, tag_type)
        tokens_per_sentence = TaggedCorpus._get_tokens_per_sentence(sentences)
        label_size_dict = {

        }
        for (l, c) in classes_to_count.items():
            label_size_dict[l] = c
        tag_size_dict = {

        }
        for (l, c) in tags_to_count.items():
            tag_size_dict[l] = c
        return {
            u'dataset': name,
            u'total_number_of_documents': len(sentences),
            u'number_of_documents_per_class': label_size_dict,
            u'number_of_tokens_per_tag': tag_size_dict,
            u'number_of_tokens': {
                u'total': sum(tokens_per_sentence),
                u'min': min(tokens_per_sentence),
                u'max': max(tokens_per_sentence),
                u'avg': (sum(tokens_per_sentence) / len(sentences)),
            },
        }

    @staticmethod
    def _get_tokens_per_sentence(sentences):
        return list(map((lambda x: len(x.tokens)), sentences))

    @staticmethod
    def _get_class_to_count(sentences):
        class_to_count = defaultdict((lambda: 0))
        for sent in sentences:
            for label in sent.labels:
                class_to_count[label.value] += 1
        return class_to_count

    @staticmethod
    def _get_tag_to_count(sentences, tag_type):
        tag_to_count = defaultdict((lambda: 0))
        for sent in sentences:
            for word in sent.tokens:
                if (tag_type in word.tags):
                    label = word.tags[tag_type]
                    tag_to_count[label.value] += 1
        return tag_to_count

    def __str__(self):
        return (u'TaggedCorpus: %d train + %d dev + %d test sentences' % (len(self.train), len(self.dev), len(self.test)))


def iob2(tags):
    u'\n    Check that tags have a valid IOB format.\n    Tags in IOB1 format are converted to IOB2.\n    '
    for (i, tag) in enumerate(tags):
        if (tag.value == u'O'):
            continue
        split = tag.value.split(u'-')
        if ((len(split) != 2) or (split[0] not in [u'I', u'B'])):
            return False
        if (split[0] == u'B'):
            continue
        elif ((i == 0) or (tags[(i - 1)].value == u'O')):
            tags[i].value = (u'B' + tag.value[1:])
        elif (tags[(i - 1)].value[1:] == tag.value[1:]):
            continue
        else:
            tags[i].value = (u'B' + tag.value[1:])
    return True


def iob_iobes(tags):
    u'\n    IOB -> IOBES\n    '
    new_tags = []
    for (i, tag) in enumerate(tags):
        if (tag.value == u'O'):
            new_tags.append(tag.value)
        elif (tag.value.split(u'-')[0] == u'B'):
            if (((i + 1) != len(tags)) and (tags[(i + 1)].value.split(u'-')[0] == u'I')):
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace(u'B-', u'S-'))
        elif (tag.value.split(u'-')[0] == u'I'):
            if (((i + 1) < len(tags)) and (tags[(i + 1)].value.split(u'-')[0] == u'I')):
                new_tags.append(tag.value)
            else:
                new_tags.append(tag.value.replace(u'I-', u'E-'))
        else:
            raise Exception(u'Invalid IOB format!')
    return new_tags


class MultiCorpus(Corpus):

    def __init__(self, corpora):
        self.corpora = corpora

    @property
    def train(self):
        train = []
        for corpus in self.corpora:
            train.extend(corpus.train)
        return train

    @property
    def dev(self):
        dev = []
        for corpus in self.corpora:
            dev.extend(corpus.dev)
        return dev

    @property
    def test(self):
        test = []
        for corpus in self.corpora:
            test.extend(corpus.test)
        return test

    def __str__(self):
        return u'\n'.join([unicode(corpus) for corpus in self.corpora])

    def get_all_sentences(self):
        sentences = []
        for corpus in self.corpora:
            sentences.extend(corpus.get_all_sentences())
        return sentences

    def downsample(self, percentage=0.1, only_downsample_train=False):
        for corpus in self.corpora:
            corpus.downsample(percentage, only_downsample_train)
        return self

    def make_tag_dictionary(self, tag_type):
        tag_dictionary = Dictionary()
        tag_dictionary.add_item(u'O')
        for corpus in self.corpora:
            for sentence in corpus.get_all_sentences():
                for token in sentence.tokens:
                    token = token
                    tag_dictionary.add_item(token.get_tag(tag_type).value)
        tag_dictionary.add_item(u'<START>')
        tag_dictionary.add_item(u'<STOP>')
        return tag_dictionary

    def make_label_dictionary(self):
        label_dictionary = Dictionary(add_unk=False)
        for corpus in self.corpora:
            labels = set(corpus._get_all_label_names())
            for label in labels:
                label_dictionary.add_item(label)
        return label_dictionary
