
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import shutil
import pytest
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from flair.data import Dictionary, Sentence
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings, TokenEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger, TextClassifier, LanguageModel
from flair.trainers import ModelTrainer
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.training_utils import EvaluationMetric
from flair.optim import AdamW


@pytest.mark.integration
def test_train_load_use_tagger(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = WordEmbeddings(u'glove')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.1, mini_batch_size=2, max_epochs=2, test_mode=True)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_load_use_tagger_large(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.UD_ENGLISH).downsample(0.05)
    tag_dictionary = corpus.make_tag_dictionary(u'pos')
    embeddings = WordEmbeddings(u'glove')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'pos', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.1, mini_batch_size=32, max_epochs=2, test_mode=True)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_load_use_tagger(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = FlairEmbeddings(u'news-forward-fast')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.1, mini_batch_size=2, max_epochs=2, test_mode=True)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_changed_chache_load_use_tagger(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    cache_dir = (results_base_path / u'cache')
    os.makedirs(cache_dir, exist_ok=True)
    embeddings = FlairEmbeddings(
        u'news-forward-fast', cache_directory=cache_dir)
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, EvaluationMetric.MACRO_ACCURACY,
                  learning_rate=0.1, mini_batch_size=2, max_epochs=2, test_mode=True)
    shutil.rmtree(cache_dir)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_nochache_load_use_tagger(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = FlairEmbeddings(u'news-forward-fast', use_cache=False)
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, test_mode=True)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_optimizer(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = WordEmbeddings(u'glove')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    optimizer = Adam
    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE,
                  learning_rate=0.1, mini_batch_size=2, max_epochs=2, test_mode=True)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_optimizer_arguments(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = WordEmbeddings(u'glove')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    optimizer = AdamW
    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, test_mode=True, weight_decay=0.001)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_find_learning_rate(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(
        NLPTask.FASHION, base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = WordEmbeddings(u'glove')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    optimizer = SGD
    trainer = ModelTrainer(tagger, corpus, optimizer=optimizer)
    trainer.find_learning_rate(results_base_path, iterations=5)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_load_use_serialized_tagger():
    loaded_model = SequenceTagger.load(u'ner')
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    sentence.clear_embeddings()
    sentence_empty.clear_embeddings()
    loaded_model = SequenceTagger.load(u'pos')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])


@pytest.mark.integration
def test_train_load_use_classifier(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(u'imdb', base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()
    glove_embedding = WordEmbeddings(u'en-glove')
    document_embeddings = DocumentLSTMEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE,
                  max_epochs=2, test_mode=True)
    sentence = Sentence(u'Berlin is a really nice city.')
    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_load_use_classifier_multi_label(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_classification_corpus(
        data_folder=(tasks_base_path / u'multi_class'))
    label_dict = corpus.make_label_dictionary()
    glove_embedding = WordEmbeddings(u'en-glove')
    document_embeddings = DocumentLSTMEmbeddings(embeddings=[
                                                 glove_embedding], hidden_size=32, reproject_words=False, bidirectional=False)
    model = TextClassifier(document_embeddings, label_dict, multi_label=True)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, EvaluationMetric.MICRO_F1_SCORE,
                  max_epochs=100, test_mode=True, checkpoint=False)
    sentence = Sentence(u'apple tv')
    for s in model.predict(sentence):
        for l in s.labels:
            print(l)
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    sentence = Sentence(u'apple tv')
    for s in model.predict(sentence):
        assert (u'apple' in sentence.get_label_names())
        assert (u'tv' in sentence.get_label_names())
        for l in s.labels:
            print(l)
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_load_use_classifier(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(u'imdb', base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()
    glove_embedding = FlairEmbeddings(u'news-forward-fast')
    document_embeddings = DocumentLSTMEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, EvaluationMetric.MACRO_F1_SCORE,
                  max_epochs=2, test_mode=True)
    sentence = Sentence(u'Berlin is a really nice city.')
    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_charlm_nocache_load_use_classifier(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(u'imdb', base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()
    glove_embedding = FlairEmbeddings(u'news-forward-fast', use_cache=False)
    document_embeddings = DocumentLSTMEmbeddings(
        [glove_embedding], 128, 1, False, 64, False, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2, test_mode=True)
    sentence = Sentence(u'Berlin is a really nice city.')
    for s in model.predict(sentence):
        for l in s.labels:
            assert (l.value is not None)
            assert (0.0 <= l.score <= 1.0)
            assert (type(l.score) is float)
    loaded_model = TextClassifier.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_language_model(results_base_path, resources_path):
    dictionary = Dictionary.load(u'chars')
    language_model = LanguageModel(
        dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)
    corpus = TextCorpus((resources_path / u'corpora/lorem_ipsum'),
                        dictionary, language_model.is_forward_lm, character_level=True)
    trainer = LanguageModelTrainer(language_model, corpus, test_mode=True)
    trainer.train(results_base_path, sequence_length=10,
                  mini_batch_size=10, max_epochs=2)
    char_lm_embeddings = FlairEmbeddings(
        unicode((results_base_path / u'best-lm.pt')))
    sentence = Sentence(u'I love Berlin')
    char_lm_embeddings.embed(sentence)
    (text, likelihood) = language_model.generate_text(number_of_characters=100)
    assert (text is not None)
    assert (len(text) >= 100)
    shutil.rmtree(results_base_path, ignore_errors=True)


@pytest.mark.integration
def test_train_load_use_tagger_multicorpus(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpora(
        [NLPTask.FASHION, NLPTask.GERMEVAL], base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = WordEmbeddings(u'glove')
    tagger = SequenceTagger(hidden_size=64, embeddings=embeddings,
                            tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(results_base_path, learning_rate=0.1,
                  mini_batch_size=2, max_epochs=2, test_mode=True)
    loaded_model = SequenceTagger.load_from_file(
        (results_base_path / u'final-model.pt'))
    sentence = Sentence(u'I love Berlin')
    sentence_empty = Sentence(u'       ')
    loaded_model.predict(sentence)
    loaded_model.predict([sentence, sentence_empty])
    loaded_model.predict([sentence_empty])
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_resume_text_classification_training(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpus(u'imdb', base_path=tasks_base_path)
    label_dict = corpus.make_label_dictionary()
    embeddings = FlairEmbeddings(u'news-forward-fast', use_cache=False)
    document_embeddings = DocumentLSTMEmbeddings([embeddings], 128, 1, False)
    model = TextClassifier(document_embeddings, label_dict, False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2,
                  test_mode=True, checkpoint=True)
    trainer = ModelTrainer.load_from_checkpoint(
        (results_base_path / u'checkpoint.pt'), u'TextClassifier', corpus)
    trainer.train(results_base_path, max_epochs=2,
                  test_mode=True, checkpoint=True)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_resume_sequence_tagging_training(results_base_path, tasks_base_path):
    corpus = NLPTaskDataFetcher.load_corpora(
        [NLPTask.FASHION, NLPTask.GERMEVAL], base_path=tasks_base_path)
    tag_dictionary = corpus.make_tag_dictionary(u'ner')
    embeddings = WordEmbeddings(u'glove')
    model = SequenceTagger(hidden_size=64, embeddings=embeddings,
                           tag_dictionary=tag_dictionary, tag_type=u'ner', use_crf=False)
    trainer = ModelTrainer(model, corpus)
    trainer.train(results_base_path, max_epochs=2,
                  test_mode=True, checkpoint=True)
    trainer = ModelTrainer.load_from_checkpoint(
        (results_base_path / u'checkpoint.pt'), u'SequenceTagger', corpus)
    trainer.train(results_base_path, max_epochs=2,
                  test_mode=True, checkpoint=True)
    shutil.rmtree(results_base_path)


@pytest.mark.integration
def test_train_resume_language_model_training(resources_path, results_base_path, tasks_base_path):
    dictionary = Dictionary.load(u'chars')
    language_model = LanguageModel(
        dictionary, is_forward_lm=True, hidden_size=128, nlayers=1)
    corpus = TextCorpus((resources_path / u'corpora/lorem_ipsum'),
                        dictionary, language_model.is_forward_lm, character_level=True)
    trainer = LanguageModelTrainer(language_model, corpus, test_mode=True)
    trainer.train(results_base_path, sequence_length=10,
                  mini_batch_size=10, max_epochs=2, checkpoint=True)
    trainer = LanguageModelTrainer.load_from_checkpoint(
        (results_base_path / u'checkpoint.pt'), corpus)
    trainer.train(results_base_path, sequence_length=10,
                  mini_batch_size=10, max_epochs=2)
    shutil.rmtree(results_base_path)
