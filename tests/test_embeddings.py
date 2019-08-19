
import pytest
from flair.embeddings import WordEmbeddings, TokenEmbeddings, StackedEmbeddings, DocumentLSTMEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings
from flair.data import Sentence


def test_loading_not_existing_embedding():
    with pytest.raises(ValueError):
        WordEmbeddings('other')
    with pytest.raises(ValueError):
        WordEmbeddings('not/existing/path/to/embeddings')


def test_loading_not_existing_char_lm_embedding():
    with pytest.raises(ValueError):
        FlairEmbeddings('other')


@pytest.mark.integration
def test_stacked_embeddings():
    (sentence, glove, charlm) = init_document_embeddings()
    embeddings = StackedEmbeddings([glove, charlm])
    embeddings.embed(sentence)
    for token in sentence.tokens:
        assert (len(token.get_embedding()) == 1124)
        token.clear_embeddings()
        assert (len(token.get_embedding()) == 0)


@pytest.mark.integration
def test_document_lstm_embeddings():
    (sentence, glove, charlm) = init_document_embeddings()
    embeddings = DocumentLSTMEmbeddings(
        [glove, charlm], hidden_size=128, bidirectional=False)
    embeddings.embed(sentence)
    assert (len(sentence.get_embedding()) == 128)
    assert (len(sentence.get_embedding()) == embeddings.embedding_length)
    sentence.clear_embeddings()
    assert (len(sentence.get_embedding()) == 0)


@pytest.mark.integration
def test_document_bidirectional_lstm_embeddings():
    (sentence, glove, charlm) = init_document_embeddings()
    embeddings = DocumentLSTMEmbeddings(
        [glove, charlm], hidden_size=128, bidirectional=True)
    embeddings.embed(sentence)
    assert (len(sentence.get_embedding()) == 512)
    assert (len(sentence.get_embedding()) == embeddings.embedding_length)
    sentence.clear_embeddings()
    assert (len(sentence.get_embedding()) == 0)


@pytest.mark.integration
def test_document_pool_embeddings():
    (sentence, glove, charlm) = init_document_embeddings()
    for mode in ['mean', 'max', 'min']:
        embeddings = DocumentPoolEmbeddings([glove, charlm], mode=mode)
        embeddings.embed(sentence)
        assert (len(sentence.get_embedding()) == 1124)
        sentence.clear_embeddings()
        assert (len(sentence.get_embedding()) == 0)


def init_document_embeddings():
    text = 'I love Berlin. Berlin is a great place to live.'
    sentence = Sentence(text)
    glove = WordEmbeddings('en-glove')
    charlm = FlairEmbeddings('news-forward-fast')
    return (sentence, glove, charlm)


def load_and_apply_word_embeddings(emb_type: str):
    text = 'I love Berlin.'
    sentence = Sentence(text)
    embeddings = WordEmbeddings(emb_type)
    embeddings.embed(sentence)
    for token in sentence.tokens:
        assert (len(token.get_embedding()) != 0)
        token.clear_embeddings()
        assert (len(token.get_embedding()) == 0)


def load_and_apply_char_lm_embeddings(emb_type: str):
    text = 'I love Berlin.'
    sentence = Sentence(text)
    embeddings = FlairEmbeddings(emb_type)
    embeddings.embed(sentence)
    for token in sentence.tokens:
        assert (len(token.get_embedding()) != 0)
        token.clear_embeddings()
        assert (len(token.get_embedding()) == 0)
