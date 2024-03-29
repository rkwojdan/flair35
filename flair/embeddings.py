
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import logging
from abc import abstractmethod
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
from typing import List, Union, Dict
import gensim
import numpy as np
import torch
from bpemb import BPEmb
from deprecated import deprecated
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, PRETRAINED_MODEL_ARCHIVE_MAP
import flair
from .nn import LockedDropout, WordDropout
from .data import Dictionary, Token, Sentence
from .file_utils import cached_path
log = logging.getLogger(u'flair')


class Embeddings(torch.nn.Module):
    u'Abstract base class for all embeddings. Every new type of embedding must implement these methods.'

    @property
    @abstractmethod
    def embedding_length(self):
        u'Returns the length of the embedding vector.'
        pass

    @property
    @abstractmethod
    def embedding_type(self):
        pass

    def embed(self, sentences):
        u'Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings\n\n        are non-static.'
        if (type(sentences) is Sentence):
            sentences = [sentences]
        everything_embedded = True
        if (self.embedding_type == u'word-level'):
            for sentence in sentences:
                for token in sentence.tokens:
                    if (self.name not in token._embeddings.keys()):
                        everything_embedded = False
        else:
            for sentence in sentences:
                if (self.name not in sentence._embeddings.keys()):
                    everything_embedded = False
        if ((not everything_embedded) or (not self.static_embeddings)):
            self._add_embeddings_internal(sentences)
        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences):
        u'Private method for adding embeddings to all words in a list of sentences.'
        pass


class TokenEmbeddings(Embeddings):
    u'Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods.'

    @property
    @abstractmethod
    def embedding_length(self):
        u'Returns the length of the embedding vector.'
        pass

    @property
    def embedding_type(self):
        return u'word-level'


class DocumentEmbeddings(Embeddings):
    u'Abstract base class for all document-level embeddings. Ever new type of document embedding must implement these methods.'

    @property
    @abstractmethod
    def embedding_length(self):
        u'Returns the length of the embedding vector.'
        pass

    @property
    def embedding_type(self):
        return u'sentence-level'


class StackedEmbeddings(TokenEmbeddings):
    u'A stack of embeddings, used if you need to combine several different embedding types.'

    def __init__(self, embeddings, detach=True):
        u'The constructor takes a list of embeddings to be combined.'
        super(StackedEmbeddings, self).__init__()
        self.embeddings = embeddings
        for (i, embedding) in enumerate(embeddings):
            self.add_module(u'list_embedding_{}'.format(i), embedding)
        self.detach = detach
        self.name = u'Stack'
        self.static_embeddings = True
        self.__embedding_type = embeddings[0].embedding_type
        self.__embedding_length = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, sentences, static_embeddings=True):
        if (type(sentences) is Sentence):
            sentences = [sentences]
        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self):
        return self.__embedding_type

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)
        return sentences

    def __str__(self):
        return u''.join([u'StackedEmbeddings [', u'{}'.format(u','.join([unicode(e) for e in self.embeddings])), u']'])


class WordEmbeddings(TokenEmbeddings):
    u'Standard static word embeddings, such as GloVe or FastText.'

    def __init__(self, embeddings, field=None):
        u"\n\n        Initializes classic word embeddings. Constructor downloads required files if not there.\n\n        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code.\n\n        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.\n\n        "
        old_base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/'
        base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/'
        embeddings_path_v4 = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/'
        cache_dir = Path(u'embeddings')
        if ((embeddings.lower() == u'glove') or (embeddings.lower() == u'en-glove')):
            cached_path(u''.join(
                [u'{}'.format(old_base_path), u'glove.gensim.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(
                u''.join([u'{}'.format(old_base_path), u'glove.gensim']), cache_dir=cache_dir)
        elif ((embeddings.lower() == u'extvec') or (embeddings.lower() == u'en-extvec')):
            cached_path(u''.join(
                [u'{}'.format(old_base_path), u'extvec.gensim.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(
                u''.join([u'{}'.format(old_base_path), u'extvec.gensim']), cache_dir=cache_dir)
        elif ((embeddings.lower() == u'crawl') or (embeddings.lower() == u'en-crawl')):
            cached_path(u''.join([u'{}'.format(
                base_path), u'en-fasttext-crawl-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(u''.join(
                [u'{}'.format(base_path), u'en-fasttext-crawl-300d-1M']), cache_dir=cache_dir)
        elif ((embeddings.lower() == u'news') or (embeddings.lower() == u'en-news') or (embeddings.lower() == u'en')):
            cached_path(u''.join([u'{}'.format(
                base_path), u'en-fasttext-news-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(u''.join(
                [u'{}'.format(base_path), u'en-fasttext-news-300d-1M']), cache_dir=cache_dir)
        elif ((embeddings.lower() == u'twitter') or (embeddings.lower() == u'en-twitter')):
            cached_path(u''.join([u'{}'.format(
                old_base_path), u'twitter.gensim.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(u''.join(
                [u'{}'.format(old_base_path), u'twitter.gensim']), cache_dir=cache_dir)
        elif (len(embeddings.lower()) == 2):
            cached_path(u''.join([u'{}'.format(embeddings_path_v4), u'{}'.format(
                embeddings), u'-wiki-fasttext-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(u''.join([u'{}'.format(embeddings_path_v4), u'{}'.format(
                embeddings), u'-wiki-fasttext-300d-1M']), cache_dir=cache_dir)
        elif ((len(embeddings.lower()) == 7) and embeddings.endswith(u'-wiki')):
            cached_path(u''.join([u'{}'.format(embeddings_path_v4), u'{}'.format(
                embeddings[:2]), u'-wiki-fasttext-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(u''.join([u'{}'.format(embeddings_path_v4), u'{}'.format(
                embeddings[:2]), u'-wiki-fasttext-300d-1M']), cache_dir=cache_dir)
        elif ((len(embeddings.lower()) == 8) and embeddings.endswith(u'-crawl')):
            cached_path(u''.join([u'{}'.format(embeddings_path_v4), u'{}'.format(
                embeddings[:2]), u'-crawl-fasttext-300d-1M.vectors.npy']), cache_dir=cache_dir)
            embeddings = cached_path(u''.join([u'{}'.format(embeddings_path_v4), u'{}'.format(
                embeddings[:2]), u'-crawl-fasttext-300d-1M']), cache_dir=cache_dir)
        elif (not Path(embeddings).exists()):
            raise ValueError(u''.join([u'The given embeddings "', u'{}'.format(
                embeddings), u'" is not available or is not a valid path.']))
        self.name = unicode(embeddings)
        self.static_embeddings = True
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
            unicode(embeddings))
        self.field = field
        self.__embedding_length = self.precomputed_word_embeddings.vector_size
        super(WordEmbeddings, self).__init__()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for (i, sentence) in enumerate(sentences):
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                token = token
                if ((u'field' not in self.__dict__) or (self.field is None)):
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                if (word in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[word]
                elif (word.lower() in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[word.lower(
                    )]
                elif (re.sub(u'\\d', u'#', word.lower()) in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[re.sub(
                        u'\\d', u'#', word.lower())]
                elif (re.sub(u'\\d', u'0', word.lower()) in self.precomputed_word_embeddings):
                    word_embedding = self.precomputed_word_embeddings[re.sub(
                        u'\\d', u'0', word.lower())]
                else:
                    word_embedding = np.zeros(
                        self.embedding_length, dtype=u'float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def __str__(self):
        return self.name


class BPEmbSerializable(BPEmb):

    def __getstate__(self):
        state = self.__dict__.copy()
        state[u'spm_model_binary'] = open(self.model_file, mode=u'rb').read()
        state[u'spm'] = None
        return state

    def __setstate__(self, state):
        from bpemb.util import sentencepiece_load
        model_file = self.model_tpl.format(
            lang=state[u'lang'], vs=state[u'vs'])
        self.__dict__ = state
        self.cache_dir = (Path(flair.file_utils.CACHE_ROOT) / u'embeddings')
        if (u'spm_model_binary' in self.__dict__):
            if (not os.path.exists((self.cache_dir / state[u'lang']))):
                os.makedirs((self.cache_dir / state[u'lang']))
            self.model_file = (self.cache_dir / model_file)
            with open(self.model_file, u'wb') as out:
                out.write(self.__dict__[u'spm_model_binary'])
        else:
            self.model_file = self._load_file(model_file)
        state[u'spm'] = sentencepiece_load(self.model_file)


class BytePairEmbeddings(TokenEmbeddings):

    def __init__(self, language, dim=50, syllables=100000, cache_dir=(Path(flair.file_utils.CACHE_ROOT) / u'embeddings')):
        u'\n\n        Initializes BP embeddings. Constructor downloads required files if not there.\n\n        '
        self.name = u''.join([u'bpe-', u'{}'.format(language),
                              u'-', u'{}'.format(syllables), u'-', u'{}'.format(dim)])
        self.static_embeddings = True
        self.embedder = BPEmbSerializable(
            lang=language, vs=syllables, dim=dim, cache_dir=cache_dir)
        self.__embedding_length = (self.embedder.emb.vector_size * 2)
        super(BytePairEmbeddings, self).__init__()

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for (i, sentence) in enumerate(sentences):
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                token = token
                if ((u'field' not in self.__dict__) or (self.field is None)):
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                embeddings = self.embedder.embed(word.lower())
                embedding = np.concatenate(
                    (embeddings[0], embeddings[(len(embeddings) - 1)]))
                token.set_embedding(self.name, torch.tensor(
                    embedding, dtype=torch.float))
        return sentences

    def __str__(self):
        return self.name


class ELMoEmbeddings(TokenEmbeddings):
    u'Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018.'

    def __init__(self, model=u'original'):
        super(ELMoEmbeddings, self).__init__()
        try:
            import allennlp.commands.elmo
        except:
            log.warning((u'-' * 100))
            log.warning(u'ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                u'To use ELMoEmbeddings, please first install with "pip install allennlp"')
            log.warning((u'-' * 100))
            pass
        self.name = (u'elmo-' + model)
        self.static_embeddings = True
        options_file = allennlp.commands.elmo.DEFAULT_OPTIONS_FILE
        weight_file = allennlp.commands.elmo.DEFAULT_WEIGHT_FILE
        if (model == u'small'):
            options_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
            weight_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        if (model == u'medium'):
            options_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'
            weight_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
        if ((model == u'pt') or (model == u'portuguese')):
            options_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json'
            weight_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5'
        if (model == u'pubmed'):
            options_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            weight_file = u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'
        from flair import device
        cuda_device = (0 if (unicode(device) != u'cpu') else (- 1))
        self.ee = allennlp.commands.elmo.ElmoEmbedder(
            options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token(u'hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        sentence_words = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])
        embeddings = self.ee.embed_batch(sentence_words)
        for (i, sentence) in enumerate(sentences):
            sentence_embeddings = embeddings[i]
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                token = token
                word_embedding = torch.cat([torch.FloatTensor(sentence_embeddings[0, token_idx, :]), torch.FloatTensor(
                    sentence_embeddings[1, token_idx, :]), torch.FloatTensor(sentence_embeddings[2, token_idx, :])], 0)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def extra_repr(self):
        return u'model={}'.format(self.name)

    def __str__(self):
        return self.name


class ELMoTransformerEmbeddings(TokenEmbeddings):
    u'Contextual word embeddings using word-level Transformer-based LM, as proposed in Peters et al., 2018.'

    def __init__(self, model_file):
        super(ELMoTransformerEmbeddings, self).__init__()
        try:
            from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import BidirectionalLanguageModelTokenEmbedder
            from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
        except:
            log.warning((u'-' * 100))
            log.warning(u'ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                u'To use ELMoTransformerEmbeddings, please first install a recent version from https://github.com/allenai/allennlp')
            log.warning((u'-' * 100))
            pass
        self.name = u'elmo-transformer'
        self.static_embeddings = True
        self.lm_embedder = BidirectionalLanguageModelTokenEmbedder(
            archive_file=model_file, dropout=0.2, bos_eos_tokens=(u'<S>', u'</S>'), remove_bos_eos=True, requires_grad=False)
        self.lm_embedder = self.lm_embedder.to(device=flair.device)
        self.vocab = self.lm_embedder._lm.vocab
        self.indexer = ELMoTokenCharactersIndexer()
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token(u'hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        import allennlp.data.tokenizers.token as allen_nlp_token
        indexer = self.indexer
        vocab = self.vocab
        for sentence in sentences:
            character_indices = indexer.tokens_to_indices([allen_nlp_token.Token(
                token.text) for token in sentence], vocab, u'elmo')[u'elmo']
            indices_tensor = torch.LongTensor([character_indices])
            indices_tensor = indices_tensor.to(device=flair.device)
            embeddings = self.lm_embedder(indices_tensor)[
                0].detach().cpu().numpy()
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                token = token
                embedding = embeddings[token_idx]
                word_embedding = torch.FloatTensor(embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

    def extra_repr(self):
        return u'model={}'.format(self.name)

    def __str__(self):
        return self.name


class CharacterEmbeddings(TokenEmbeddings):
    u'Character embeddings of words, as proposed in Lample et al., 2016.'

    def __init__(self, path_to_char_dict=None):
        u'Uses the default character dictionary if none provided.'
        super(CharacterEmbeddings, self).__init__()
        self.name = u'Char'
        self.static_embeddings = False
        if (path_to_char_dict is None):
            self.char_dictionary = Dictionary.load(u'common-chars')
        else:
            self.char_dictionary = Dictionary.load_from_file(path_to_char_dict)
        self.char_embedding_dim = 25
        self.hidden_size_char = 25
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim)
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim, self.hidden_size_char, num_layers=1, bidirectional=True)
        self.__embedding_length = (self.char_embedding_dim * 2)
        self.to(flair.device)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        for sentence in sentences:
            tokens_char_indices = []
            for token in sentence.tokens:
                token = token
                char_indices = [self.char_dictionary.get_idx_for_item(
                    char) for char in token.text]
                tokens_char_indices.append(char_indices)
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=(lambda p: len(p)), reverse=True)
            d = {

            }
            for (i, ci) in enumerate(tokens_char_indices):
                for (j, cj) in enumerate(tokens_sorted_by_length):
                    if (ci == cj):
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence), dtype=torch.long, device=flair.device)
            for (i, c) in enumerate(tokens_sorted_by_length):
                tokens_mask[i, :chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device)
            chars = tokens_mask
            character_embeddings = self.char_embedding(chars).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length)
            (lstm_out, self.hidden) = self.char_rnn(packed)
            (outputs, output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)), dtype=torch.float, device=flair.device)
            for (i, index) in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[(i, (index - 1))]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]
            for (token_number, token) in enumerate(sentence.tokens):
                token.set_embedding(
                    self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


class FlairEmbeddings(TokenEmbeddings):
    u'Contextual string embeddings of words, as proposed in Akbik et al., 2018.'

    def __init__(self, model, use_cache=False, cache_directory=None, chars_per_chunk=512):
        u"\n\n        initializes contextual string embeddings using a character-level language model.\n\n        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',\n\n                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'\n\n                depending on which character language model is desired.\n\n        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will\n\n                not allow re-use of once computed embeddings that do not fit into memory\n\n        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache\n\n                is written to the provided directory.\n\n        :param  chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster but requires\n\n                more memory. Lower means slower but less memory.\n\n        "
        super(FlairEmbeddings, self).__init__()
        cache_dir = Path(u'embeddings')
        if (model.lower() == u'multi-forward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'multi-backward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'multi-forward-fast'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-fast-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'multi-backward-fast'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-fast-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-forward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-backward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-forward-fast'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-backward-fast'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'mix-forward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'mix-backward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'german-forward') or (model.lower() == u'de-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'german-backward') or (model.lower() == u'de-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'polish-forward') or (model.lower() == u'pl-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-forward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'polish-backward') or (model.lower() == u'pl-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-backward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'slovenian-forward') or (model.lower() == u'sl-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'slovenian-backward') or (model.lower() == u'sl-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'bulgarian-forward') or (model.lower() == u'bg-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'bulgarian-backward') or (model.lower() == u'bg-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'dutch-forward') or (model.lower() == u'nl-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'dutch-backward') or (model.lower() == u'nl-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'swedish-forward') or (model.lower() == u'sv-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'swedish-backward') or (model.lower() == u'sv-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'french-forward') or (model.lower() == u'fr-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'french-backward') or (model.lower() == u'fr-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'czech-forward') or (model.lower() == u'cs-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'czech-backward') or (model.lower() == u'cs-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'portuguese-forward') or (model.lower() == u'pt-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'portuguese-backward') or (model.lower() == u'pt-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'basque-forward') or (model.lower() == u'eu-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-eu-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'basque-backward') or (model.lower() == u'eu-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-eu-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'spanish-forward-fast') or (model.lower() == u'es-forward-fast')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/language_model_es_forward/lm-es-forward-fast.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'spanish-backward-fast') or (model.lower() == u'es-backward-fast')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/language_model_es_backward/lm-es-backward-fast.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'spanish-forward') or (model.lower() == u'es-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/language_model_es_forward_long/lm-es-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'spanish-backward') or (model.lower() == u'es-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/language_model_es_backward_long/lm-es-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (not Path(model).exists()):
            raise ValueError(u''.join([u'The given model "', u'{}'.format(
                model), u'" is not available or is not a valid path.']))
        self.name = unicode(model)
        self.static_embeddings = True
        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.is_forward_lm = self.lm.is_forward_lm
        self.chars_per_chunk = chars_per_chunk
        self.cache = None
        if use_cache:
            cache_path = (Path(u''.join([u'{}'.format(self.name), u'-tmp-cache.sqllite'])) if (
                not cache_directory) else (cache_directory / u''.join([u'{}'.format(self.name), u'-tmp-cache.sqllite'])))
            from sqlitedict import SqliteDict
            self.cache = SqliteDict(unicode(cache_path), autocommit=True)
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token(u'hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state[u'cache'] = None
        return state

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        if (u'chars_per_chunk' not in self.__dict__):
            self.chars_per_chunk = 512
        if ((u'cache' in self.__dict__) and (self.cache is not None)):
            all_embeddings_retrieved_from_cache = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)
                if (not embeddings):
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for (token, embedding) in zip(sentence, embeddings):
                        token.set_embedding(
                            self.name, torch.FloatTensor(embedding))
            if all_embeddings_retrieved_from_cache:
                return sentences
        with torch.no_grad():
            text_sentences = [sentence.to_tokenized_string()
                              for sentence in sentences]
            longest_character_sequence_in_batch = len(
                max(text_sentences, key=len))
            sentences_padded = []
            append_padded_sentence = sentences_padded.append
            start_marker = u'\n'
            end_marker = u' '
            extra_offset = len(start_marker)
            for sentence_text in text_sentences:
                pad_by = (longest_character_sequence_in_batch -
                          len(sentence_text))
                if self.is_forward_lm:
                    padded = u'{}{}{}{}'.format(
                        start_marker, sentence_text, end_marker, (pad_by * u' '))
                    append_padded_sentence(padded)
                else:
                    padded = u'{}{}{}{}'.format(
                        start_marker, sentence_text[::(- 1)], end_marker, (pad_by * u' '))
                    append_padded_sentence(padded)
            all_hidden_states_in_lm = self.lm.get_representation(
                sentences_padded, self.chars_per_chunk)
            for (i, sentence) in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string()
                offset_forward = extra_offset
                offset_backward = (len(sentence_text) + extra_offset)
                for token in sentence.tokens:
                    token = token
                    offset_forward += len(token.text)
                    if self.is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward
                    embedding = all_hidden_states_in_lm[offset, i, :]
                    offset_forward += 1
                    offset_backward -= 1
                    offset_backward -= len(token.text)
                    token.set_embedding(self.name, embedding.clone().detach())
            all_hidden_states_in_lm = None
        if ((u'cache' in self.__dict__) and (self.cache is not None)):
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string()] = [
                    token._embeddings[self.name].tolist() for token in sentence]
        return sentences

    def __str__(self):
        return self.name


class PooledFlairEmbeddings(TokenEmbeddings):

    def __init__(self, contextual_embeddings, pooling=u'fade', only_capitalized=False, **kwargs):
        super(PooledFlairEmbeddings, self).__init__()
        if (type(contextual_embeddings) is unicode):
            self.context_embeddings = FlairEmbeddings(
                contextual_embeddings, **kwargs)
        else:
            self.context_embeddings = contextual_embeddings
        self.embedding_length = (self.context_embeddings.embedding_length * 2)
        self.name = (self.context_embeddings.name + u'-context')
        self.word_embeddings = {

        }
        self.word_count = {

        }
        self.only_capitalized = only_capitalized
        self.static_embeddings = False
        self.pooling = pooling
        if (pooling == u'mean'):
            self.aggregate_op = torch.add
        elif (pooling == u'fade'):
            self.aggregate_op = torch.add
        elif (pooling == u'max'):
            self.aggregate_op = torch.max
        elif (pooling == u'min'):
            self.aggregate_op = torch.min

    def train(self, mode=True):
        super(PooledFlairEmbeddings, self).train(mode=mode)
        if mode:
            print(u'train mode resetting embeddings')
            self.word_embeddings = {

            }
            self.word_count = {

            }

    def _add_embeddings_internal(self, sentences):
        self.context_embeddings.embed(sentences)
        for sentence in sentences:
            for token in sentence.tokens:
                local_embedding = token._embeddings[self.context_embeddings.name]
                if (token.text[0].isupper() or (not self.only_capitalized)):
                    if (token.text not in self.word_embeddings):
                        self.word_embeddings[token.text] = local_embedding
                        self.word_count[token.text] = 1
                    else:
                        aggregated_embedding = self.aggregate_op(
                            self.word_embeddings[token.text], local_embedding)
                        if (self.pooling == u'fade'):
                            aggregated_embedding /= 2
                        self.word_embeddings[token.text] = aggregated_embedding
                        self.word_count[token.text] += 1
        for sentence in sentences:
            for token in sentence.tokens:
                if (token.text in self.word_embeddings):
                    base = ((self.word_embeddings[token.text] / self.word_count[token.text]) if (
                        self.pooling == u'mean') else self.word_embeddings[token.text])
                else:
                    base = token._embeddings[self.context_embeddings.name]
                token.set_embedding(self.name, base)
        return sentences

    def embedding_length(self):
        return self.embedding_length


class BertEmbeddings(TokenEmbeddings):

    def __init__(self, bert_model=u'bert-base-uncased', layers=u'-1,-2,-3,-4', pooling_operation=u'first'):
        u"\n\n        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.\n\n        :param bert_model: name of BERT model ('')\n\n        :param layers: string indicating which layers to take for embedding\n\n        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take\n\n        the average ('mean') or use first word piece embedding as token embedding ('first)\n\n        "
        super(BertEmbeddings, self).__init__()
        if (bert_model not in PRETRAINED_MODEL_ARCHIVE_MAP.keys()):
            raise ValueError(u'Provided bert-model is not available.')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertModel.from_pretrained(bert_model)
        self.layer_indexes = [int(x) for x in layers.split(u',')]
        self.pooling_operation = pooling_operation
        self.name = unicode(bert_model)
        self.static_embeddings = True

    class BertInputFeatures(object):
        u'Private helper class for holding BERT-formatted features'

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, token_subtoken_count):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(self, sentences, max_sequence_length):
        max_sequence_length = (max_sequence_length + 2)
        features = []
        for (sentence_index, sentence) in enumerate(sentences):
            bert_tokenization = []
            token_subtoken_count = {

            }
            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)
            if (len(bert_tokenization) > (max_sequence_length - 2)):
                bert_tokenization = bert_tokenization[0:(
                    max_sequence_length - 2)]
            tokens = []
            input_type_ids = []
            tokens.append(u'[CLS]')
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append(u'[SEP]')
            input_type_ids.append(0)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = ([1] * len(input_ids))
            while (len(input_ids) < max_sequence_length):
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)
            features.append(BertEmbeddings.BertInputFeatures(unique_id=sentence_index, tokens=tokens, input_ids=input_ids,
                                                             input_mask=input_mask, input_type_ids=input_type_ids, token_subtoken_count=token_subtoken_count))
        return features

    def _add_embeddings_internal(self, sentences):
        u'Add embeddings to all words in a list of sentences. If embeddings are already added,\n\n        updates only if embeddings are non-static.'
        longest_sentence_in_batch = len(max([self.tokenizer.tokenize(
            sentence.to_tokenized_string()) for sentence in sentences], key=len))
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch)
        all_input_ids = torch.LongTensor(
            [f.input_ids for f in features]).to(flair.device)
        all_input_masks = torch.LongTensor(
            [f.input_mask for f in features]).to(flair.device)
        self.model.to(flair.device)
        self.model.eval()
        (all_encoder_layers, _) = self.model(all_input_ids,
                                             token_type_ids=None, attention_mask=all_input_masks)
        with torch.no_grad():
            for (sentence_index, sentence) in enumerate(sentences):
                feature = features[sentence_index]
                subtoken_embeddings = []
                for (token_index, _) in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu()[
                            sentence_index]
                        all_layers.append(layer_output[token_index])
                    subtoken_embeddings.append(torch.cat(all_layers))
                token_idx = 0
                for token in sentence:
                    token_idx += 1
                    if (self.pooling_operation == u'first'):
                        token.set_embedding(
                            self.name, subtoken_embeddings[token_idx])
                    else:
                        embeddings = subtoken_embeddings[token_idx:(
                            token_idx + feature.token_subtoken_count[token.idx])]
                        embeddings = [embedding.unsqueeze(
                            0) for embedding in embeddings]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)
                    token_idx += (feature.token_subtoken_count[token.idx] - 1)
        return sentences

    @property
    @abstractmethod
    def embedding_length(self):
        u'Returns the length of the embedding vector.'
        return (len(self.layer_indexes) * self.model.config.hidden_size)


class CharLMEmbeddings(TokenEmbeddings):
    u'Contextual string embeddings of words, as proposed in Akbik et al., 2018. '

    @deprecated(version=u'0.4', reason=u"Use 'FlairEmbeddings' instead.")
    def __init__(self, model, detach=True, use_cache=False, cache_directory=None):
        u"\n\n        initializes contextual string embeddings using a character-level language model.\n\n        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',\n\n                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward'\n\n                depending on which character language model is desired.\n\n        :param detach: if set to False, the gradient will propagate into the language model. this dramatically slows down\n\n                training and often leads to worse results, so not recommended.\n\n        :param use_cache: if set to False, will not write embeddings to file for later retrieval. this saves disk space but will\n\n                not allow re-use of once computed embeddings that do not fit into memory\n\n        :param cache_directory: if cache_directory is not set, the cache will be written to ~/.flair/embeddings. otherwise the cache\n\n                is written to the provided directory.\n\n        "
        super(CharLMEmbeddings, self).__init__()
        cache_dir = Path(u'embeddings')
        if (model.lower() == u'multi-forward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'multi-backward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-multi-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-forward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-backward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-forward-fast'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-forward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'news-backward-fast'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-news-english-backward-1024-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'mix-forward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'mix-backward'):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-english-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'german-forward') or (model.lower() == u'de-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-forward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'german-backward') or (model.lower() == u'de-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-mix-german-backward-v0.2rc.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'polish-forward') or (model.lower() == u'pl-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-forward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'polish-backward') or (model.lower() == u'pl-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-polish-backward-v0.2.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'slovenian-forward') or (model.lower() == u'sl-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'slovenian-backward') or (model.lower() == u'sl-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-sl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'bulgarian-forward') or (model.lower() == u'bg-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'bulgarian-backward') or (model.lower() == u'bg-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/lm-bg-small-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'dutch-forward') or (model.lower() == u'nl-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'dutch-backward') or (model.lower() == u'nl-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-nl-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'swedish-forward') or (model.lower() == u'sv-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'swedish-backward') or (model.lower() == u'sv-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-sv-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'french-forward') or (model.lower() == u'fr-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'french-backward') or (model.lower() == u'fr-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/lm-fr-charlm-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'czech-forward') or (model.lower() == u'cs-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-forward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'czech-backward') or (model.lower() == u'cs-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-cs-large-backward-v0.1.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'portuguese-forward') or (model.lower() == u'pt-forward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-forward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'portuguese-backward') or (model.lower() == u'pt-backward')):
            base_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/lm-pt-backward.pt'
            model = cached_path(base_path, cache_dir=cache_dir)
        elif (not Path(model).exists()):
            raise ValueError(u''.join([u'The given model "', u'{}'.format(
                model), u'" is not available or is not a valid path.']))
        self.name = unicode(model)
        self.static_embeddings = detach
        from flair.models import LanguageModel
        self.lm = LanguageModel.load_language_model(model)
        self.detach = detach
        self.is_forward_lm = self.lm.is_forward_lm
        self.cache = None
        if use_cache:
            cache_path = (Path(u''.join([u'{}'.format(self.name), u'-tmp-cache.sqllite'])) if (
                not cache_directory) else (cache_directory / u''.join([u'{}'.format(self.name), u'-tmp-cache.sqllite'])))
            from sqlitedict import SqliteDict
            self.cache = SqliteDict(unicode(cache_path), autocommit=True)
        dummy_sentence = Sentence()
        dummy_sentence.add_token(Token(u'hello'))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length = len(
            embedded_dummy[0].get_token(1).get_embedding())
        self.eval()

    def train(self, mode=True):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state[u'cache'] = None
        return state

    @property
    def embedding_length(self):
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):
        if ((u'cache' in self.__dict__) and (self.cache is not None)):
            all_embeddings_retrieved_from_cache = True
            for sentence in sentences:
                key = sentence.to_tokenized_string()
                embeddings = self.cache.get(key)
                if (not embeddings):
                    all_embeddings_retrieved_from_cache = False
                    break
                else:
                    for (token, embedding) in zip(sentence, embeddings):
                        token.set_embedding(
                            self.name, torch.FloatTensor(embedding))
            if all_embeddings_retrieved_from_cache:
                return sentences
        text_sentences = [sentence.to_tokenized_string()
                          for sentence in sentences]
        longest_character_sequence_in_batch = len(max(text_sentences, key=len))
        sentences_padded = []
        append_padded_sentence = sentences_padded.append
        end_marker = u' '
        extra_offset = 1
        for sentence_text in text_sentences:
            pad_by = (longest_character_sequence_in_batch - len(sentence_text))
            if self.is_forward_lm:
                padded = u'\n{}{}{}'.format(
                    sentence_text, end_marker, (pad_by * u' '))
                append_padded_sentence(padded)
            else:
                padded = u'\n{}{}{}'.format(
                    sentence_text[::(- 1)], end_marker, (pad_by * u' '))
                append_padded_sentence(padded)
        all_hidden_states_in_lm = self.lm.get_representation(sentences_padded)
        for (i, sentence) in enumerate(sentences):
            sentence_text = sentence.to_tokenized_string()
            offset_forward = extra_offset
            offset_backward = (len(sentence_text) + extra_offset)
            for token in sentence.tokens:
                token = token
                offset_forward += len(token.text)
                if self.is_forward_lm:
                    offset = offset_forward
                else:
                    offset = offset_backward
                embedding = all_hidden_states_in_lm[offset, i, :]
                offset_forward += 1
                offset_backward -= 1
                offset_backward -= len(token.text)
                token.set_embedding(self.name, embedding)
        if ((u'cache' in self.__dict__) and (self.cache is not None)):
            for sentence in sentences:
                self.cache[sentence.to_tokenized_string()] = [
                    token._embeddings[self.name].tolist() for token in sentence]
        return sentences

    def __str__(self):
        return self.name


class DocumentMeanEmbeddings(DocumentEmbeddings):

    @deprecated(version=u'0.3.1', reason=u"The functionality of this class is moved to 'DocumentPoolEmbeddings'")
    def __init__(self, token_embeddings):
        u'The constructor takes a list of embeddings to be combined.'
        super(DocumentMeanEmbeddings, self).__init__()
        self.embeddings = StackedEmbeddings(embeddings=token_embeddings)
        self.name = u'document_mean'
        self.__embedding_length = self.embeddings.embedding_length
        self.to(flair.device)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        u'Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates\n\n        only if embeddings are non-static.'
        everything_embedded = True
        if (type(sentences) is Sentence):
            sentences = [sentences]
        for sentence in sentences:
            if (self.name not in sentence._embeddings.keys()):
                everything_embedded = False
        if (not everything_embedded):
            self.embeddings.embed(sentences)
            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    token = token
                    word_embeddings.append(token.get_embedding().unsqueeze(0))
                word_embeddings = torch.cat(
                    word_embeddings, dim=0).to(flair.device)
                mean_embedding = torch.mean(word_embeddings, 0)
                sentence.set_embedding(self.name, mean_embedding)

    def _add_embeddings_internal(self, sentences):
        pass


class DocumentPoolEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings, mode=u'mean'):
        u"The constructor takes a list of embeddings to be combined.\n\n        :param embeddings: a list of token embeddings\n\n        :param mode: a string which can any value from ['mean', 'max', 'min']\n\n        "
        super(DocumentPoolEmbeddings, self).__init__()
        self.embeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length
        self.to(flair.device)
        self.mode = mode
        if (self.mode == u'mean'):
            self.pool_op = torch.mean
        elif (mode == u'max'):
            self.pool_op = torch.max
        elif (mode == u'min'):
            self.pool_op = torch.min
        else:
            raise ValueError(u''.join(
                [u'Pooling operation for ', u'{}'.format(self.mode), u' is not defined']))
        self.name = u''.join([u'document_', u'{}'.format(self.mode)])

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        u'Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates\n\n        only if embeddings are non-static.'
        everything_embedded = True
        if isinstance(sentences, Sentence):
            sentences = [sentences]
        for sentence in sentences:
            if (self.name not in sentence._embeddings.keys()):
                everything_embedded = False
        if (not everything_embedded):
            self.embeddings.embed(sentences)
            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    token = token
                    word_embeddings.append(token.get_embedding().unsqueeze(0))
                word_embeddings = torch.cat(
                    word_embeddings, dim=0).to(flair.device)
                if (self.mode == u'mean'):
                    pooled_embedding = self.pool_op(word_embeddings, 0)
                else:
                    (pooled_embedding, _) = self.pool_op(word_embeddings, 0)
                sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences):
        pass


class DocumentLSTMEmbeddings(DocumentEmbeddings):

    def __init__(self, embeddings, hidden_size=128, rnn_layers=1, reproject_words=True, reproject_words_dimension=None, bidirectional=False, dropout=0.5, word_dropout=0.0, locked_dropout=0.0):
        u'The constructor takes a list of embeddings to be combined.\n\n        :param embeddings: a list of token embeddings\n\n        :param hidden_size: the number of hidden states in the lstm\n\n        :param rnn_layers: the number of layers for the lstm\n\n        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear\n\n        layer before putting them into the lstm or not\n\n        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output\n\n        dimension as before will be taken.\n\n        :param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not\n\n        :param dropout: the dropout value to be used\n\n        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used\n\n        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used\n\n        '
        super(DocumentLSTMEmbeddings, self).__init__()
        self.embeddings = StackedEmbeddings(embeddings=embeddings)
        self.reproject_words = reproject_words
        self.bidirectional = bidirectional
        self.length_of_all_token_embeddings = self.embeddings.embedding_length
        self.name = u'document_lstm'
        self.static_embeddings = False
        self.__embedding_length = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4
        self.embeddings_dimension = self.length_of_all_token_embeddings
        if (self.reproject_words and (reproject_words_dimension is not None)):
            self.embeddings_dimension = reproject_words_dimension
        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension)
        self.rnn = torch.nn.GRU(self.embeddings_dimension, hidden_size,
                                num_layers=rnn_layers, bidirectional=self.bidirectional)
        if (locked_dropout > 0.0):
            self.dropout = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)
        self.use_word_dropout = (word_dropout > 0.0)
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)
        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)
        self.to(flair.device)

    @property
    def embedding_length(self):
        return self.__embedding_length

    def embed(self, sentences):
        u'Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update\n\n         only if embeddings are non-static.'
        if (type(sentences) is Sentence):
            sentences = [sentences]
        self.rnn.zero_grad()
        sentences.sort(key=(lambda x: len(x)), reverse=True)
        self.embeddings.embed(sentences)
        longest_token_sequence_in_batch = len(sentences[0])
        all_sentence_tensors = []
        lengths = []
        for (i, sentence) in enumerate(sentences):
            lengths.append(len(sentence.tokens))
            word_embeddings = []
            for (token, token_idx) in zip(sentence.tokens, range(len(sentence.tokens))):
                token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))
            for add in range((longest_token_sequence_in_batch - len(sentence.tokens))):
                word_embeddings.append(torch.zeros(
                    self.length_of_all_token_embeddings, dtype=torch.float).unsqueeze(0))
            word_embeddings_tensor = torch.cat(
                word_embeddings, 0).to(flair.device)
            sentence_states = word_embeddings_tensor
            all_sentence_tensors.append(sentence_states.unsqueeze(1))
        sentence_tensor = torch.cat(all_sentence_tensors, 1)
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        sentence_tensor = self.dropout(sentence_tensor)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths)
        self.rnn.flatten_parameters()
        (lstm_out, hidden) = self.rnn(packed)
        (outputs, output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        outputs = self.dropout(outputs)
        for (sentence_no, length) in enumerate(lengths):
            last_rep = outputs[((length - 1), sentence_no)]
            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[(0, sentence_no)]
                embedding = torch.cat([first_rep, last_rep], 0)
            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences):
        pass


class DocumentLMEmbeddings(DocumentEmbeddings):

    def __init__(self, flair_embeddings, detach=True):
        super(DocumentLMEmbeddings, self).__init__()
        self.embeddings = flair_embeddings
        self.name = u'document_lm'
        self.static_embeddings = detach
        self.detach = detach
        self._embedding_length = sum(
            (embedding.embedding_length for embedding in flair_embeddings))

    @property
    def embedding_length(self):
        return self._embedding_length

    def _add_embeddings_internal(self, sentences):
        if (type(sentences) is Sentence):
            sentences = [sentences]
        for embedding in self.embeddings:
            embedding.embed(sentences)
            for sentence in sentences:
                if embedding.is_forward_lm:
                    sentence.set_embedding(
                        embedding.name, sentence[len(sentence)]._embeddings[embedding.name])
                else:
                    sentence.set_embedding(
                        embedding.name, sentence[1]._embeddings[embedding.name])
        return sentences
