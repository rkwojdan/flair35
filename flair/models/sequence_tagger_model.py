
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import warnings
import logging
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import torch.nn
from torch.optim import Optimizer
import torch.nn.functional as F
import flair.nn
import torch
import flair.embeddings
from flair.data import Dictionary, Sentence, Token, Label
from flair.file_utils import cached_path
from typing import List, Tuple, Union
from flair.training_utils import clear_embeddings
from tqdm import tqdm
log = logging.getLogger(u'flair')
START_TAG = u'<START>'
STOP_TAG = u'<STOP>'


def to_scalar(var):
    return var.view((- 1)).detach().tolist()[0]


def argmax(vec):
    (_, idx) = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[(0, argmax(vec))]
    max_score_broadcast = max_score.view(1, (- 1)).expand(1, vec.size()[1])
    return (max_score + torch.log(torch.sum(torch.exp((vec - max_score_broadcast)))))


def argmax_batch(vecs):
    (_, idx) = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp((vecs - maxi_bc)), 1))
    return (maxi + recti_)


def pad_tensors(tensor_list):
    ml = max([x.shape[0] for x in tensor_list])
    shape = ([len(tensor_list), ml] + list(tensor_list[0].shape[1:]))
    template = torch.zeros(*list(shape), dtype=torch.long, device=flair.device)
    lens_ = [x.shape[0] for x in tensor_list]
    for (i, tensor) in enumerate(tensor_list):
        template[i, :lens_[i]] = tensor
    return (template, lens_)


class SequenceTagger(flair.nn.Model):

    def __init__(self, hidden_size, embeddings, tag_dictionary, tag_type, use_crf=True, use_rnn=True, rnn_layers=1, dropout=0.0, word_dropout=0.05, locked_dropout=0.5, pickle_module=u'pickle'):
        super(SequenceTagger, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf = use_crf
        self.rnn_layers = rnn_layers
        self.trained_epochs = 0
        self.embeddings = embeddings
        self.tag_dictionary = tag_dictionary
        self.tag_type = tag_type
        self.tagset_size = len(tag_dictionary)
        self.nlayers = rnn_layers
        self.hidden_word = None
        self.use_dropout = dropout
        self.use_word_dropout = word_dropout
        self.use_locked_dropout = locked_dropout
        self.pickle_module = pickle_module
        if (dropout > 0.0):
            self.dropout = torch.nn.Dropout(dropout)
        if (word_dropout > 0.0):
            self.word_dropout = flair.nn.WordDropout(word_dropout)
        if (locked_dropout > 0.0):
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
        rnn_input_dim = self.embeddings.embedding_length
        self.relearn_embeddings = True
        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)
        self.rnn_type = u'LSTM'
        if (self.rnn_type in [u'LSTM', u'GRU']):
            if (self.nlayers == 1):
                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim, hidden_size, num_layers=self.nlayers, bidirectional=True)
            else:
                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim, hidden_size, num_layers=self.nlayers, dropout=0.5, bidirectional=True)
        if self.use_rnn:
            self.linear = torch.nn.Linear(
                (hidden_size * 2), len(tag_dictionary))
        else:
            self.linear = torch.nn.Linear(
                self.embeddings.embedding_length, len(tag_dictionary))
        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.detach()[self.tag_dictionary.get_idx_for_item(
                START_TAG), :] = (- 10000)
            self.transitions.detach(
            )[:, self.tag_dictionary.get_idx_for_item(STOP_TAG)] = (- 10000)
        self.to(flair.device)

    @staticmethod
    def save_torch_model(model_state, model_file, pickle_module=u'pickle', pickle_protocol=4):
        if (pickle_module == u'dill'):
            try:
                import dill
                torch.save(model_state, unicode(
                    model_file), pickle_module=dill)
            except:
                log.warning((u'-' * 100))
                log.warning(u'ATTENTION! The library "dill" is not installed!')
                log.warning(
                    u'Please first install "dill" with "pip install dill" to save the model!')
                log.warning((u'-' * 100))
                pass
        else:
            torch.save(model_state, unicode(model_file),
                       pickle_protocol=pickle_protocol)

    def save(self, model_file):
        model_state = {
            u'state_dict': self.state_dict(),
            u'embeddings': self.embeddings,
            u'hidden_size': self.hidden_size,
            u'tag_dictionary': self.tag_dictionary,
            u'tag_type': self.tag_type,
            u'use_crf': self.use_crf,
            u'use_rnn': self.use_rnn,
            u'rnn_layers': self.rnn_layers,
            u'use_word_dropout': self.use_word_dropout,
            u'use_locked_dropout': self.use_locked_dropout,
        }
        self.save_torch_model(model_state, unicode(
            model_file), self.pickle_module)

    def save_checkpoint(self, model_file, optimizer_state, scheduler_state, epoch, loss):
        model_state = {
            u'state_dict': self.state_dict(),
            u'embeddings': self.embeddings,
            u'hidden_size': self.hidden_size,
            u'tag_dictionary': self.tag_dictionary,
            u'tag_type': self.tag_type,
            u'use_crf': self.use_crf,
            u'use_rnn': self.use_rnn,
            u'rnn_layers': self.rnn_layers,
            u'use_word_dropout': self.use_word_dropout,
            u'use_locked_dropout': self.use_locked_dropout,
            u'optimizer_state_dict': optimizer_state,
            u'scheduler_state_dict': scheduler_state,
            u'epoch': epoch,
            u'loss': loss,
        }
        self.save_torch_model(model_state, unicode(
            model_file), self.pickle_module)

    @classmethod
    def load_from_file(cls, model_file):
        state = SequenceTagger._load_state(model_file)
        use_dropout = (0.0 if (not (u'use_dropout' in state.keys()))
                       else state[u'use_dropout'])
        use_word_dropout = (0.0 if (
            not (u'use_word_dropout' in state.keys())) else state[u'use_word_dropout'])
        use_locked_dropout = (0.0 if (
            not (u'use_locked_dropout' in state.keys())) else state[u'use_locked_dropout'])
        model = SequenceTagger(hidden_size=state[u'hidden_size'], embeddings=state[u'embeddings'], tag_dictionary=state[u'tag_dictionary'], tag_type=state[u'tag_type'],
                               use_crf=state[u'use_crf'], use_rnn=state[u'use_rnn'], rnn_layers=state[u'rnn_layers'], dropout=use_dropout, word_dropout=use_word_dropout, locked_dropout=use_locked_dropout)
        model.load_state_dict(state[u'state_dict'])
        model.eval()
        model.to(flair.device)
        return model

    @classmethod
    def load_checkpoint(cls, model_file):
        state = SequenceTagger._load_state(model_file)
        model = SequenceTagger.load_from_file(model_file)
        epoch = (state[u'epoch'] if (u'epoch' in state) else None)
        loss = (state[u'loss'] if (u'loss' in state) else None)
        optimizer_state_dict = (state[u'optimizer_state_dict'] if (
            u'optimizer_state_dict' in state) else None)
        scheduler_state_dict = (state[u'scheduler_state_dict'] if (
            u'scheduler_state_dict' in state) else None)
        return {
            u'model': model,
            u'epoch': epoch,
            u'loss': loss,
            u'optimizer_state_dict': optimizer_state_dict,
            u'scheduler_state_dict': scheduler_state_dict,
        }

    @classmethod
    def _load_state(cls, model_file):
        with warnings.catch_warnings():
            warnings.filterwarnings(u'ignore')
            f = flair.file_utils.load_big_file(unicode(model_file))
            state = torch.load(f, map_location=flair.device)
            return state

    def forward_loss(self, sentences, sort=True):
        (features, lengths, tags) = self.forward(sentences, sort=sort)
        return self._calculate_loss(features, lengths, tags)

    def forward_labels_and_loss(self, sentences, sort=True):
        with torch.no_grad():
            (feature, lengths, tags) = self.forward(sentences, sort=sort)
            loss = self._calculate_loss(feature, lengths, tags)
            tags = self._obtain_labels(feature, lengths)
            return (tags, loss)

    def predict(self, sentences, mini_batch_size=32, verbose=False):
        with torch.no_grad():
            if isinstance(sentences, Sentence):
                sentences = [sentences]
            filtered_sentences = self._filter_empty_sentences(sentences)
            clear_embeddings(filtered_sentences,
                             also_clear_word_embeddings=True)
            filtered_sentences.sort(key=(lambda x: len(x)), reverse=True)
            batches = [filtered_sentences[x:(
                x + mini_batch_size)] for x in range(0, len(filtered_sentences), mini_batch_size)]
            if verbose:
                batches = tqdm(batches)
            for (i, batch) in enumerate(batches):
                if verbose:
                    batches.set_description(
                        u''.join([u'Inferencing on batch ', u'{}'.format(i)]))
                (tags, _) = self.forward_labels_and_loss(batch, sort=False)
                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token = token
                        token.add_tag_label(self.tag_type, tag)
                clear_embeddings(batch, also_clear_word_embeddings=True)
            return sentences

    def forward(self, sentences, sort=True):
        self.zero_grad()
        self.embeddings.embed(sentences)
        if sort:
            sentences.sort(key=(lambda x: len(x)), reverse=True)
        lengths = [len(sentence.tokens) for sentence in sentences]
        tag_list = []
        longest_token_sequence_in_batch = lengths[0]
        sentence_tensor = torch.zeros([len(sentences), longest_token_sequence_in_batch,
                                       self.embeddings.embedding_length], dtype=torch.float, device=flair.device)
        for (s_id, sentence) in enumerate(sentences):
            sentence_tensor[s_id][:len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0)
            tag_idx = [self.tag_dictionary.get_idx_for_item(
                token.get_tag(self.tag_type).value) for token in sentence]
            tag = torch.LongTensor(tag_idx).to(flair.device)
            tag_list.append(tag)
        sentence_tensor = sentence_tensor.transpose_(0, 1)
        if (self.use_dropout > 0.0):
            sentence_tensor = self.dropout(sentence_tensor)
        if (self.use_word_dropout > 0.0):
            sentence_tensor = self.word_dropout(sentence_tensor)
        if (self.use_locked_dropout > 0.0):
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)
        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths)
            (rnn_output, hidden) = self.rnn(packed)
            (sentence_tensor, output_lengths) = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output)
            if (self.use_dropout > 0.0):
                sentence_tensor = self.dropout(sentence_tensor)
            if (self.use_locked_dropout > 0.0):
                sentence_tensor = self.locked_dropout(sentence_tensor)
        features = self.linear(sentence_tensor)
        return (features.transpose_(0, 1), lengths, tag_list)

    def _score_sentence(self, feats, tags, lens_):
        start = torch.LongTensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)]).to(flair.device)
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = torch.LongTensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)]).to(flair.device)
        stop = stop[None, :].repeat(tags.shape[0], 1)
        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]                          :] = self.tag_dictionary.get_idx_for_item(STOP_TAG)
        score = torch.FloatTensor(feats.shape[0]).to(flair.device)
        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(flair.device)
            score[i] = (torch.sum(self.transitions[(pad_stop_tags[i, :(lens_[i] + 1)],
                                                    pad_start_tags[i, :(lens_[i] + 1)])]) + torch.sum(feats[(i, r, tags[i, :lens_[i]])]))
        return score

    def _calculate_loss(self, features, lengths, tags):
        if self.use_crf:
            (tags, _) = pad_tensors(tags)
            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)
            score = (forward_score - gold_score)
            return score.sum()
        else:
            score = 0
            for (sentence_feats, sentence_tags, sentence_length) in zip(features, tags, lengths):
                sentence_feats = sentence_feats[:sentence_length]
                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags)
            return score

    def _obtain_labels(self, feature, lengths):
        tags = []
        for (feats, length) in zip(feature, lengths):
            if self.use_crf:
                (confidences, tag_seq) = self._viterbi_decode(feats[:length])
            else:
                tag_seq = []
                confidences = []
                for backscore in feats[:length]:
                    softmax = F.softmax(backscore, dim=0)
                    (_, idx) = torch.max(backscore, 0)
                    prediction = idx.item()
                    tag_seq.append(prediction)
                    confidences.append(softmax[prediction].item())
            tags.append([Label(self.tag_dictionary.get_item_for_index(tag), conf)
                         for (conf, tag) in zip(confidences, tag_seq)])
        return tags

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        init_vvars = torch.FloatTensor(1, self.tagset_size).to(
            flair.device).fill_((- 10000.0))
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = init_vvars
        for feat in feats:
            next_tag_var = (forward_var.view(
                1, (- 1)).expand(self.tagset_size, self.tagset_size) + self.transitions)
            (_, bptrs_t) = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[(range(len(bptrs_t)), bptrs_t)]
            forward_var = (viterbivars_t + feat)
            backscores.append(forward_var)
            backpointers.append(bptrs_t)
        terminal_var = (
            forward_var + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)])
        terminal_var.detach()[self.tag_dictionary.get_idx_for_item(
            STOP_TAG)] = (- 10000.0)
        terminal_var.detach()[self.tag_dictionary.get_idx_for_item(
            START_TAG)] = (- 10000.0)
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            (_, idx) = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
        start = best_path.pop()
        assert (start == self.tag_dictionary.get_idx_for_item(START_TAG))
        best_path.reverse()
        return (best_scores, best_path)

    def _forward_alg(self, feats, lens_):
        init_alphas = torch.FloatTensor(self.tagset_size).fill_((- 10000.0))
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0
        forward_var = torch.zeros(
            feats.shape[0], (feats.shape[1] + 1), feats.shape[2], dtype=torch.float, device=flair.device)
        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]).repeat(feats.shape[0], 1, 1)
        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = ((emit_score[:, :, None].repeat(1, 1, transitions.shape[2]) + transitions) +
                       forward_var[:, i, :][:, :, None].repeat(1, 1, transitions.shape[2]).transpose(2, 1))
            (max_tag_var, _) = torch.max(tag_var, dim=2)
            tag_var = (
                tag_var - max_tag_var[:, :, None].repeat(1, 1, transitions.shape[2]))
            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))
            cloned = forward_var.clone()
            cloned[:, (i + 1), :] = (max_tag_var + agg_)
            forward_var = cloned
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = (forward_var + self.transitions[self.tag_dictionary.get_idx_for_item(
            STOP_TAG)][None, :].repeat(forward_var.shape[0], 1))
        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    @staticmethod
    def _filter_empty_sentences(sentences):
        filtered_sentences = [
            sentence for sentence in sentences if sentence.tokens]
        if (len(sentences) != len(filtered_sentences)):
            log.warning(u'Ignore {} sentence(s) with no tokens.'.format(
                (len(sentences) - len(filtered_sentences))))
        return filtered_sentences

    @staticmethod
    def load(model):
        model_file = None
        aws_resource_path = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.2'
        aws_resource_path_v04 = u'https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4'
        cache_dir = Path(u'models')
        if ((model.lower() == u'ner-multi') or (model.lower() == u'multi-ner')):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'release-quadner-512-l2-multi-embed', u'quadner-large.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        if ((model.lower() == u'ner-multi-fast') or (model.lower() == u'multi-ner-fast')):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'NER-multi-fast', u'ner-multi-fast.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        if ((model.lower() == u'ner-multi-fast-learn') or (model.lower() == u'multi-ner-fast-learn')):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'NER-multi-fast-evolve', u'ner-multi-fast-learn.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        if (model.lower() == u'ner'):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'NER-conll03-english', u'en-ner-conll03-v0.4.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'ner-fast'):
            base_path = u'/'.join([aws_resource_path,
                                   u'NER-conll03--h256-l1-b32-experimental--fast-v0.2', u'en-ner-fast-conll03-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'ner-ontonotes'):
            base_path = u'/'.join([aws_resource_path,
                                   u'NER-ontoner--h256-l1-b32-%2Bcrawl%2Bnews-forward%2Bnews-backward--v0.2', u'en-ner-ontonotes-v0.3.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'ner-ontonotes-fast'):
            base_path = u'/'.join([aws_resource_path, u'NER-ontoner--h256-l1-b32-%2Bcrawl%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                   u'en-ner-ontonotes-fast-v0.3.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'pos-multi') or (model.lower() == u'multi-pos')):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'release-dodekapos-512-l2-multi', u'pos-multi-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif ((model.lower() == u'pos-multi-fast') or (model.lower() == u'multi-pos-fast')):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'UPOS-multi-fast', u'pos-multi-fast.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'pos'):
            base_path = u'/'.join([aws_resource_path,
                                   u'POS-ontonotes--h256-l1-b32-%2Bmix-forward%2Bmix-backward--v0.2', u'en-pos-ontonotes-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'pos-fast'):
            base_path = u'/'.join([aws_resource_path, u'POS-ontonotes--h256-l1-b32-%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                   u'en-pos-ontonotes-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'frame'):
            base_path = u'/'.join([aws_resource_path,
                                   u'FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward%2Bnews-backward--v0.2', u'en-frame-ontonotes-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'frame-fast'):
            base_path = u'/'.join([aws_resource_path, u'FRAME-conll12--h256-l1-b8-%2Bnews%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                   u'en-frame-ontonotes-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'chunk'):
            base_path = u'/'.join([aws_resource_path,
                                   u'NP-conll2000--h256-l1-b32-%2Bnews-forward%2Bnews-backward--v0.2', u'en-chunk-conll2000-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'chunk-fast'):
            base_path = u'/'.join([aws_resource_path, u'NP-conll2000--h256-l1-b32-%2Bnews-forward-fast%2Bnews-backward-fast--v0.2',
                                   u'en-chunk-conll2000-fast-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'de-pos'):
            base_path = u'/'.join([aws_resource_path,
                                   u'UPOS-udgerman--h256-l1-b8-%2Bgerman-forward%2Bgerman-backward--v0.2', u'de-pos-ud-v0.2.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'de-pos-fine-grained'):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'POS-fine-grained-german-tweets', u'de-pos-twitter-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'de-ner'):
            base_path = u'/'.join([aws_resource_path,
                                   u'NER-conll03ger--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--v0.2', u'de-ner-conll03-v0.3.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'de-ner-germeval'):
            base_path = u'/'.join([aws_resource_path,
                                   u'NER-germeval--h256-l1-b32-%2Bde-fasttext%2Bgerman-forward%2Bgerman-backward--v0.2', u'de-ner-germeval-v0.3.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'fr-ner'):
            base_path = u'/'.join([aws_resource_path,
                                   u'NER-aij-wikiner-fr-wp3', u'fr-ner.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        elif (model.lower() == u'nl-ner'):
            base_path = u'/'.join([aws_resource_path_v04,
                                   u'NER-conll2002-dutch', u'nl-ner-conll02-v0.1.pt'])
            model_file = cached_path(base_path, cache_dir=cache_dir)
        if (model_file is not None):
            tagger = SequenceTagger.load_from_file(model_file)
            return tagger
