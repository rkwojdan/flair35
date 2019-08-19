
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import pytest
from typing import List
from flair.data import Sentence, Label, Token, Dictionary, TaggedCorpus, Span
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask


def test_get_head():
    token1 = Token(u'I', 0)
    token2 = Token(u'love', 1, 0)
    token3 = Token(u'Berlin', 2, 1)
    sentence = Sentence()
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)
    assert (token2 == token3.get_head())
    assert (token1 == token2.get_head())
    assert (None == token1.get_head())


def test_create_sentence_without_tokenizer():
    sentence = Sentence(u'I love Berlin.')
    assert (3 == len(sentence.tokens))
    assert (u'I' == sentence.tokens[0].text)
    assert (u'love' == sentence.tokens[1].text)
    assert (u'Berlin.' == sentence.tokens[2].text)


def test_token_indices():
    text = u':    nation on'
    sentence = Sentence(text)
    assert (text == sentence.to_original_text())
    text = u':    nation on'
    sentence = Sentence(text, use_tokenizer=True)
    assert (text == sentence.to_original_text())
    text = u'I love Berlin.'
    sentence = Sentence(text)
    assert (text == sentence.to_original_text())
    text = u'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
    sentence = Sentence(text)
    assert (text == sentence.to_original_text())
    text = u'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .'
    sentence = Sentence(text, use_tokenizer=True)
    assert (text == sentence.to_original_text())


def test_create_sentence_with_tokenizer():
    sentence = Sentence(u'I love Berlin.', use_tokenizer=True)
    assert (4 == len(sentence.tokens))
    assert (u'I' == sentence.tokens[0].text)
    assert (u'love' == sentence.tokens[1].text)
    assert (u'Berlin' == sentence.tokens[2].text)
    assert (u'.' == sentence.tokens[3].text)


def test_sentence_to_plain_string():
    sentence = Sentence(u'I love Berlin.', use_tokenizer=True)
    assert (u'I love Berlin .' == sentence.to_tokenized_string())


def test_sentence_to_real_string(tasks_base_path):
    sentence = Sentence(u'I love Berlin.', use_tokenizer=True)
    assert (u'I love Berlin.' == sentence.to_plain_string())
    corpus = NLPTaskDataFetcher.load_corpus(NLPTask.GERMEVAL, tasks_base_path)
    sentence = corpus.train[0]
    assert (u'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in einer Weise aufgetreten , die alles andere als überzeugend war " .' == sentence.to_tokenized_string())
    assert (u'Schartau sagte dem "Tagesspiegel" vom Freitag, Fischer sei "in einer Weise aufgetreten, die alles andere als überzeugend war".' == sentence.to_plain_string())
    sentence = corpus.train[1]
    assert (u'Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als Möbelvertreter , als er einen fliegenden Händler aus dem Libanon traf .' == sentence.to_tokenized_string())
    assert (u'Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als Möbelvertreter, als er einen fliegenden Händler aus dem Libanon traf.' == sentence.to_plain_string())


def test_sentence_infer_tokenization():
    sentence = Sentence()
    sentence.add_token(Token(u'xyz'))
    sentence.add_token(Token(u'"'))
    sentence.add_token(Token(u'abc'))
    sentence.add_token(Token(u'"'))
    sentence.infer_space_after()
    assert (u'xyz " abc "' == sentence.to_tokenized_string())
    assert (u'xyz "abc"' == sentence.to_plain_string())
    sentence = Sentence(u'xyz " abc "')
    sentence.infer_space_after()
    assert (u'xyz " abc "' == sentence.to_tokenized_string())
    assert (u'xyz "abc"' == sentence.to_plain_string())


def test_sentence_get_item():
    sentence = Sentence(u'I love Berlin.', use_tokenizer=True)
    assert (sentence.get_token(1) == sentence[0])
    assert (sentence.get_token(3) == sentence[2])
    with pytest.raises(IndexError):
        token = sentence[4]


def test_sentence_whitespace_tokenization():
    sentence = Sentence(u'I  love Berlin .')
    assert (4 == len(sentence.tokens))
    assert (u'I' == sentence.get_token(1).text)
    assert (u'love' == sentence.get_token(2).text)
    assert (u'Berlin' == sentence.get_token(3).text)
    assert (u'.' == sentence.get_token(4).text)


def test_sentence_to_tagged_string():
    token1 = Token(u'I', 0)
    token2 = Token(u'love', 1, 0)
    token3 = Token(u'Berlin', 2, 1)
    token3.add_tag(u'ner', u'LOC')
    sentence = Sentence()
    sentence.add_token(token1)
    sentence.add_token(token2)
    sentence.add_token(token3)
    assert (u'I love Berlin <LOC>' == sentence.to_tagged_string())


def test_dictionary_get_items_with_unk():
    dictionary = Dictionary()
    dictionary.add_item(u'class_1')
    dictionary.add_item(u'class_2')
    dictionary.add_item(u'class_3')
    items = dictionary.get_items()
    assert (4 == len(items))
    assert (u'<unk>' == items[0])
    assert (u'class_1' == items[1])
    assert (u'class_2' == items[2])
    assert (u'class_3' == items[3])


def test_dictionary_get_items_without_unk():
    dictionary = Dictionary(add_unk=False)
    dictionary.add_item(u'class_1')
    dictionary.add_item(u'class_2')
    dictionary.add_item(u'class_3')
    items = dictionary.get_items()
    assert (3 == len(items))
    assert (u'class_1' == items[0])
    assert (u'class_2' == items[1])
    assert (u'class_3' == items[2])


def test_dictionary_get_idx_for_item():
    dictionary = Dictionary(add_unk=False)
    dictionary.add_item(u'class_1')
    dictionary.add_item(u'class_2')
    dictionary.add_item(u'class_3')
    idx = dictionary.get_idx_for_item(u'class_2')
    assert (1 == idx)


def test_dictionary_get_item_for_index():
    dictionary = Dictionary(add_unk=False)
    dictionary.add_item(u'class_1')
    dictionary.add_item(u'class_2')
    dictionary.add_item(u'class_3')
    item = dictionary.get_item_for_index(0)
    assert (u'class_1' == item)


def test_dictionary_save_and_load():
    dictionary = Dictionary(add_unk=False)
    dictionary.add_item(u'class_1')
    dictionary.add_item(u'class_2')
    dictionary.add_item(u'class_3')
    file_path = u'dictionary.txt'
    dictionary.save(file_path)
    loaded_dictionary = dictionary.load_from_file(file_path)
    assert (len(dictionary) == len(loaded_dictionary))
    assert (len(dictionary.get_items()) == len(loaded_dictionary.get_items()))
    os.remove(file_path)


def test_tagged_corpus_get_all_sentences():
    train_sentence = Sentence(u"I'm used in training.", use_tokenizer=True)
    dev_sentence = Sentence(u"I'm a dev sentence.", use_tokenizer=True)
    test_sentence = Sentence(
        u'I will be only used for testing.', use_tokenizer=True)
    corpus = TaggedCorpus([train_sentence], [dev_sentence], [test_sentence])
    all_sentences = corpus.get_all_sentences()
    assert (3 == len(all_sentences))


def test_tagged_corpus_make_vocab_dictionary():
    train_sentence = Sentence(
        u'used in training. training is cool.', use_tokenizer=True)
    corpus = TaggedCorpus([train_sentence], [], [])
    vocab = corpus.make_vocab_dictionary(max_tokens=2, min_freq=(- 1))
    assert (3 == len(vocab))
    assert (u'<unk>' in vocab.get_items())
    assert (u'training' in vocab.get_items())
    assert (u'.' in vocab.get_items())
    vocab = corpus.make_vocab_dictionary(max_tokens=(- 1), min_freq=(- 1))
    assert (7 == len(vocab))
    vocab = corpus.make_vocab_dictionary(max_tokens=(- 1), min_freq=2)
    assert (3 == len(vocab))
    assert (u'<unk>' in vocab.get_items())
    assert (u'training' in vocab.get_items())
    assert (u'.' in vocab.get_items())


def test_label_set_confidence():
    label = Label(u'class_1', 3.2)
    assert (1.0 == label.score)
    assert (u'class_1' == label.value)
    label.score = 0.2
    assert (0.2 == label.score)


def test_tagged_corpus_make_label_dictionary():
    sentence_1 = Sentence(u'sentence 1', labels=[Label(u'class_1')])
    sentence_2 = Sentence(u'sentence 2', labels=[Label(u'class_2')])
    sentence_3 = Sentence(u'sentence 3', labels=[Label(u'class_1')])
    corpus = TaggedCorpus([sentence_1, sentence_2, sentence_3], [], [])
    label_dict = corpus.make_label_dictionary()
    assert (2 == len(label_dict))
    assert (u'<unk>' not in label_dict.get_items())
    assert (u'class_1' in label_dict.get_items())
    assert (u'class_2' in label_dict.get_items())


def test_tagged_corpus_make_label_dictionary_string():
    sentence_1 = Sentence(u'sentence 1', labels=[u'class_1'])
    sentence_2 = Sentence(u'sentence 2', labels=[u'class_2'])
    sentence_3 = Sentence(u'sentence 3', labels=[u'class_1'])
    corpus = TaggedCorpus([sentence_1, sentence_2, sentence_3], [], [])
    label_dict = corpus.make_label_dictionary()
    assert (2 == len(label_dict))
    assert (u'<unk>' not in label_dict.get_items())
    assert (u'class_1' in label_dict.get_items())
    assert (u'class_2' in label_dict.get_items())


def test_tagged_corpus_statistics():
    train_sentence = Sentence(u'I love Berlin.', labels=[
                              Label(u'class_1')], use_tokenizer=True)
    dev_sentence = Sentence(u'The sun is shining.', labels=[
                            Label(u'class_2')], use_tokenizer=True)
    test_sentence = Sentence(u'Berlin is sunny.', labels=[
                             Label(u'class_1')], use_tokenizer=True)
    class_to_count_dict = TaggedCorpus._get_class_to_count(
        [train_sentence, dev_sentence, test_sentence])
    assert (u'class_1' in class_to_count_dict)
    assert (u'class_2' in class_to_count_dict)
    assert (2 == class_to_count_dict[u'class_1'])
    assert (1 == class_to_count_dict[u'class_2'])
    tokens_in_sentences = TaggedCorpus._get_tokens_per_sentence(
        [train_sentence, dev_sentence, test_sentence])
    assert (3 == len(tokens_in_sentences))
    assert (4 == tokens_in_sentences[0])
    assert (5 == tokens_in_sentences[1])
    assert (4 == tokens_in_sentences[2])


def test_tagged_corpus_statistics_string_label():
    train_sentence = Sentence(u'I love Berlin.', labels=[
                              u'class_1'], use_tokenizer=True)
    dev_sentence = Sentence(u'The sun is shining.', labels=[
                            u'class_2'], use_tokenizer=True)
    test_sentence = Sentence(u'Berlin is sunny.', labels=[
                             u'class_1'], use_tokenizer=True)
    class_to_count_dict = TaggedCorpus._get_class_to_count(
        [train_sentence, dev_sentence, test_sentence])
    assert (u'class_1' in class_to_count_dict)
    assert (u'class_2' in class_to_count_dict)
    assert (2 == class_to_count_dict[u'class_1'])
    assert (1 == class_to_count_dict[u'class_2'])
    tokens_in_sentences = TaggedCorpus._get_tokens_per_sentence(
        [train_sentence, dev_sentence, test_sentence])
    assert (3 == len(tokens_in_sentences))
    assert (4 == tokens_in_sentences[0])
    assert (5 == tokens_in_sentences[1])
    assert (4 == tokens_in_sentences[2])


def test_tagged_corpus_statistics_multi_label():
    train_sentence = Sentence(u'I love Berlin.', labels=[
                              u'class_1'], use_tokenizer=True)
    dev_sentence = Sentence(u'The sun is shining.', labels=[
                            u'class_2'], use_tokenizer=True)
    test_sentence = Sentence(u'Berlin is sunny.', labels=[
                             u'class_1', u'class_2'], use_tokenizer=True)
    class_to_count_dict = TaggedCorpus._get_class_to_count(
        [train_sentence, dev_sentence, test_sentence])
    assert (u'class_1' in class_to_count_dict)
    assert (u'class_2' in class_to_count_dict)
    assert (2 == class_to_count_dict[u'class_1'])
    assert (2 == class_to_count_dict[u'class_2'])
    tokens_in_sentences = TaggedCorpus._get_tokens_per_sentence(
        [train_sentence, dev_sentence, test_sentence])
    assert (3 == len(tokens_in_sentences))
    assert (4 == tokens_in_sentences[0])
    assert (5 == tokens_in_sentences[1])
    assert (4 == tokens_in_sentences[2])


def test_tagged_corpus_get_tag_statistic():
    train_sentence = Sentence(u'Zalando Research is located in Berlin .')
    train_sentence[0].add_tag(u'ner', u'B-ORG')
    train_sentence[1].add_tag(u'ner', u'E-ORG')
    train_sentence[5].add_tag(u'ner', u'S-LOC')
    dev_sentence = Sentence(
        u'Facebook, Inc. is a company, and Google is one as well.', use_tokenizer=True)
    dev_sentence[0].add_tag(u'ner', u'B-ORG')
    dev_sentence[1].add_tag(u'ner', u'I-ORG')
    dev_sentence[2].add_tag(u'ner', u'E-ORG')
    dev_sentence[8].add_tag(u'ner', u'S-ORG')
    test_sentence = Sentence(u'Nothing to do with companies.')
    tag_to_count_dict = TaggedCorpus._get_tag_to_count(
        [train_sentence, dev_sentence, test_sentence], u'ner')
    assert (1 == tag_to_count_dict[u'S-ORG'])
    assert (1 == tag_to_count_dict[u'S-LOC'])
    assert (2 == tag_to_count_dict[u'B-ORG'])
    assert (2 == tag_to_count_dict[u'E-ORG'])
    assert (1 == tag_to_count_dict[u'I-ORG'])


def test_tagged_corpus_downsample():
    sentence = Sentence(u'I love Berlin.', labels=[
                        Label(u'class_1')], use_tokenizer=True)
    corpus = TaggedCorpus([sentence, sentence, sentence, sentence, sentence,
                           sentence, sentence, sentence, sentence, sentence], [], [])
    assert (10 == len(corpus.train))
    corpus.downsample(percentage=0.3, only_downsample_train=True)
    assert (3 == len(corpus.train))


def test_spans():
    sentence = Sentence(u'Zalando Research is located in Berlin .')
    sentence[0].add_tag(u'ner', u'B-ORG')
    sentence[1].add_tag(u'ner', u'E-ORG')
    sentence[5].add_tag(u'ner', u'S-LOC')
    spans = sentence.get_spans(u'ner')
    assert (2 == len(spans))
    assert (u'Zalando Research' == spans[0].text)
    assert (u'ORG' == spans[0].tag)
    assert (u'Berlin' == spans[1].text)
    assert (u'LOC' == spans[1].tag)
    sentence[0].add_tag(u'ner', u'B-ORG')
    sentence[1].add_tag(u'ner', u'I-ORG')
    sentence[5].add_tag(u'ner', u'B-LOC')
    spans = sentence.get_spans(u'ner')
    assert (u'Zalando Research' == spans[0].text)
    assert (u'ORG' == spans[0].tag)
    assert (u'Berlin' == spans[1].text)
    assert (u'LOC' == spans[1].tag)
    sentence[0].add_tag(u'ner', u'I-ORG')
    sentence[1].add_tag(u'ner', u'E-ORG')
    sentence[5].add_tag(u'ner', u'I-LOC')
    spans = sentence.get_spans(u'ner')
    assert (u'Zalando Research' == spans[0].text)
    assert (u'ORG' == spans[0].tag)
    assert (u'Berlin' == spans[1].text)
    assert (u'LOC' == spans[1].tag)
    sentence[0].add_tag(u'ner', u'I-ORG')
    sentence[1].add_tag(u'ner', u'E-ORG')
    sentence[2].add_tag(u'ner', u'aux')
    sentence[3].add_tag(u'ner', u'verb')
    sentence[4].add_tag(u'ner', u'preposition')
    sentence[5].add_tag(u'ner', u'I-LOC')
    spans = sentence.get_spans(u'ner')
    assert (5 == len(spans))
    assert (u'Zalando Research' == spans[0].text)
    assert (u'ORG' == spans[0].tag)
    assert (u'Berlin' == spans[4].text)
    assert (u'LOC' == spans[4].tag)
    sentence[0].add_tag(u'ner', u'I-ORG')
    sentence[1].add_tag(u'ner', u'S-LOC')
    sentence[2].add_tag(u'ner', u'aux')
    sentence[3].add_tag(u'ner', u'B-relation')
    sentence[4].add_tag(u'ner', u'E-preposition')
    sentence[5].add_tag(u'ner', u'S-LOC')
    spans = sentence.get_spans(u'ner')
    assert (5 == len(spans))
    assert (u'Zalando' == spans[0].text)
    assert (u'ORG' == spans[0].tag)
    assert (u'Research' == spans[1].text)
    assert (u'LOC' == spans[1].tag)
    assert (u'located in' == spans[3].text)
    assert (u'relation' == spans[3].tag)
    sentence = Sentence(
        u'A woman was charged on Friday with terrorist offences after three Irish Republican Army mortar bombs were found in a Belfast house , police said . ')
    sentence[11].add_tag(u'ner', u'S-MISC')
    sentence[12].add_tag(u'ner', u'B-MISC')
    sentence[13].add_tag(u'ner', u'E-MISC')
    spans = sentence.get_spans(u'ner')
    assert (2 == len(spans))
    assert (u'Irish' == spans[0].text)
    assert (u'Republican Army' == spans[1].text)
    sentence = Sentence(u'Zalando Research is located in Berlin .')
    sentence[0].add_tag(u'ner', u'B-ORG', 1.0)
    sentence[1].add_tag(u'ner', u'E-ORG', 0.9)
    sentence[5].add_tag(u'ner', u'S-LOC', 0.5)
    spans = sentence.get_spans(u'ner', min_score=0.0)
    assert (2 == len(spans))
    assert (u'Zalando Research' == spans[0].text)
    assert (u'ORG' == spans[0].tag)
    assert (0.95 == spans[0].score)
    assert (u'Berlin' == spans[1].text)
    assert (u'LOC' == spans[1].tag)
    assert (0.5 == spans[1].score)
    spans = sentence.get_spans(u'ner', min_score=0.6)
    assert (1 == len(spans))
    spans = sentence.get_spans(u'ner', min_score=0.99)
    assert (0 == len(spans))


def test_token_position_in_sentence():
    sentence = Sentence(u'I love Berlin .')
    assert (0 == sentence.tokens[0].start_position)
    assert (1 == sentence.tokens[0].end_position)
    assert (2 == sentence.tokens[1].start_position)
    assert (6 == sentence.tokens[1].end_position)
    assert (7 == sentence.tokens[2].start_position)
    assert (13 == sentence.tokens[2].end_position)
    sentence = Sentence(u' I love  Berlin.', use_tokenizer=True)
    assert (1 == sentence.tokens[0].start_position)
    assert (2 == sentence.tokens[0].end_position)
    assert (3 == sentence.tokens[1].start_position)
    assert (7 == sentence.tokens[1].end_position)
    assert (9 == sentence.tokens[2].start_position)
    assert (15 == sentence.tokens[2].end_position)


def test_sentence_to_dict():
    sentence = Sentence(u'Zalando Research is   located in Berlin, the capital of Germany.', labels=[
                        u'business'], use_tokenizer=True)
    sentence[0].add_tag(u'ner', u'B-ORG')
    sentence[1].add_tag(u'ner', u'E-ORG')
    sentence[5].add_tag(u'ner', u'S-LOC')
    sentence[10].add_tag(u'ner', u'S-LOC')
    dict = sentence.to_dict(u'ner')
    assert (
        u'Zalando Research is   located in Berlin, the capital of Germany.' == dict[u'text'])
    assert (u'Zalando Research' == dict[u'entities'][0][u'text'])
    assert (u'Berlin' == dict[u'entities'][1][u'text'])
    assert (u'Germany' == dict[u'entities'][2][u'text'])
    assert (1 == len(dict[u'labels']))
    sentence = Sentence(
        u'Facebook, Inc. is a company, and Google is one as well.', use_tokenizer=True)
    sentence[0].add_tag(u'ner', u'B-ORG')
    sentence[1].add_tag(u'ner', u'I-ORG')
    sentence[2].add_tag(u'ner', u'E-ORG')
    sentence[8].add_tag(u'ner', u'S-ORG')
    dict = sentence.to_dict(u'ner')
    assert (
        u'Facebook, Inc. is a company, and Google is one as well.' == dict[u'text'])
    assert (u'Facebook, Inc.' == dict[u'entities'][0][u'text'])
    assert (u'Google' == dict[u'entities'][1][u'text'])
    assert (0 == len(dict[u'labels']))
