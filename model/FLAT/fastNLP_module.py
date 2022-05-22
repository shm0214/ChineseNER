from fastNLP.embeddings.embedding import TokenEmbedding
from fastNLP.core import Vocabulary
from fastNLP.io.file_utils import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path
import os
import warnings
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from fastNLP.core import logger
from utils import MyDropout
from fastNLP.embeddings.contextual_embedding import ContextualEmbedding
from fastNLP.io.file_utils import PRETRAINED_BERT_MODEL_DIR


class StaticEmbedding(TokenEmbedding):
    def __init__(self,
                 vocab: Vocabulary,
                 model_dir_or_name: str = 'en',
                 embedding_dim=-1,
                 requires_grad: bool = True,
                 init_method=None,
                 lower=False,
                 dropout=0,
                 word_dropout=0,
                 normalize=False,
                 min_freq=1,
                 **kwargs):
        super(StaticEmbedding, self).__init__(vocab,
                                              word_dropout=word_dropout,
                                              dropout=dropout)
        if embedding_dim > 0:
            model_dir_or_name = None

        if model_dir_or_name is None:
            assert embedding_dim >= 1, "The dimension of embedding should be larger than 1."
            embedding_dim = int(embedding_dim)
            model_path = None
        elif model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
        elif os.path.isfile(
                os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))
        elif os.path.isdir(
                os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = _get_file_name_base_on_postfix(
                os.path.abspath(os.path.expanduser(model_dir_or_name)), '.txt')
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        truncate_vocab = (vocab.min_freq is None
                          and min_freq > 1) or (vocab.min_freq
                                                and vocab.min_freq < min_freq)
        if truncate_vocab:
            truncated_vocab = deepcopy(vocab)
            truncated_vocab.min_freq = min_freq
            truncated_vocab.word2idx = None
            if lower:  
                lowered_word_count = defaultdict(int)
                for word, count in truncated_vocab.word_count.items():
                    lowered_word_count[word.lower()] += count
                for word in truncated_vocab.word_count.keys():
                    word_count = truncated_vocab.word_count[word]
                    if lowered_word_count[word.lower(
                    )] >= min_freq and word_count < min_freq:
                        truncated_vocab.add_word_lst(
                            [word] * (min_freq - word_count),
                            no_create_entry=truncated_vocab.
                            _is_word_no_create_entry(word))

            if kwargs.get('only_train_min_freq',
                          False) and model_dir_or_name is not None:
                for word in truncated_vocab.word_count.keys():
                    if truncated_vocab._is_word_no_create_entry(
                            word
                    ) and truncated_vocab.word_count[word] < min_freq:
                        truncated_vocab.add_word_lst(
                            [word] *
                            (min_freq - truncated_vocab.word_count[word]),
                            no_create_entry=True)
            truncated_vocab.build_vocab()
            truncated_words_to_words = torch.arange(len(vocab)).long()
            for word, index in vocab:
                truncated_words_to_words[index] = truncated_vocab.to_index(
                    word)
            logger.info(
                f"{len(vocab) - len(truncated_vocab)} out of {len(vocab)} words have frequency less than {min_freq}."
            )
            vocab = truncated_vocab

        self.only_norm_found_vector = kwargs.get('only_norm_found_vector',
                                                 False)
        if lower:
            lowered_vocab = Vocabulary(padding=vocab.padding,
                                       unknown=vocab.unknown)
            for word, index in vocab:
                if vocab._is_word_no_create_entry(word):
                    lowered_vocab.add_word(word.lower(), no_create_entry=True)
                else:
                    lowered_vocab.add_word(word.lower())  
            logger.info(
                f"All word in the vocab have been lowered. There are {len(vocab)} words, {len(lowered_vocab)} "
                f"unique lowered words.")
            if model_path:
                embedding = self._load_with_vocab(model_path,
                                                  vocab=lowered_vocab,
                                                  init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab),
                                                      embedding_dim,
                                                      init_method)
                self.register_buffer('words_to_words',
                                     torch.arange(len(vocab)).long())
            if lowered_vocab.unknown:
                unknown_idx = lowered_vocab.unknown_idx
            else:
                unknown_idx = embedding.size(0) - 1 
                self.register_buffer('words_to_words',
                                     torch.arange(len(vocab)).long())
            words_to_words = torch.full((len(vocab), ),
                                        fill_value=unknown_idx).long()
            for word, index in vocab:
                if word not in lowered_vocab:
                    word = word.lower()
                    if word not in lowered_vocab and lowered_vocab._is_word_no_create_entry(
                            word):
                        continue  
                words_to_words[index] = self.words_to_words[
                    lowered_vocab.to_index(word)]
            self.register_buffer('words_to_words', words_to_words)
            self._word_unk_index = lowered_vocab.unknown_idx 
        else:
            if model_path:
                embedding = self._load_with_vocab(model_path,
                                                  vocab=vocab,
                                                  init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab),
                                                      embedding_dim,
                                                      init_method)
                self.register_buffer('words_to_words',
                                     torch.arange(len(vocab)).long())
        if not self.only_norm_found_vector and normalize:
            embedding /= (torch.norm(embedding, dim=1, keepdim=True) + 1e-12)

        if truncate_vocab:
            for i in range(len(truncated_words_to_words)):
                index_in_truncated_vocab = truncated_words_to_words[i]
                truncated_words_to_words[i] = self.words_to_words[
                    index_in_truncated_vocab]
            del self.words_to_words
            self.register_buffer('words_to_words', truncated_words_to_words)
        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0],
                                      embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None,
                                      norm_type=2,
                                      scale_grad_by_freq=False,
                                      sparse=False,
                                      _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        self.dropout = MyDropout(dropout)

    def _randomly_init_embed(self,
                             num_embedding,
                             embedding_dim,
                             init_embed=None):

        embed = torch.zeros(num_embedding, embedding_dim)

        if init_embed is None:
            nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim),
                             np.sqrt(3 / embedding_dim))
        else:
            init_embed(embed)

        return embed

    def _load_with_vocab(self,
                         embed_filepath,
                         vocab,
                         dtype=np.float32,
                         padding='<pad>',
                         unknown='<unk>',
                         error='ignore',
                         init_method=None):

        assert isinstance(vocab,
                          Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError(
                "`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.padding_idx] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = torch.from_numpy(
                            np.fromstring(' '.join(nums),
                                          sep=' ',
                                          dtype=dtype,
                                          count=dim))
                        if self.only_norm_found_vector:
                            matrix[index] = matrix[index] / np.linalg.norm(
                                matrix[index])
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn(
                            "Error occurred at the {} line.".format(idx))
                    else:
                        logger.error(
                            "Error occurred at the {} line.".format(idx))
                        raise e
            logger.info(
                "Found {} out of {} words in the pre-training embedding.".
                format(found_count, len(vocab)))
            for word, index in vocab:
                if index not in matrix and not vocab._is_word_no_create_entry(
                        word):
                    if found_unknown: 
                        matrix[index] = matrix[vocab.unknown_idx]
                    else:
                        matrix[index] = None
            vectors = self._randomly_init_embed(len(matrix), dim, init_method)

            if vocab.unknown is None:
                unknown_idx = len(matrix)
                vectors = torch.cat((vectors, torch.zeros(1, dim)),
                                    dim=0).contiguous()
            else:
                unknown_idx = vocab.unknown_idx
            self.register_buffer(
                'words_to_words',
                torch.full((len(vocab), ), fill_value=unknown_idx).long())
            for index, (index_in_vocab, vec) in enumerate(matrix.items()):
                if vec is not None:
                    vectors[index] = vec
                self.words_to_words[index_in_vocab] = index

            return vectors

    def drop_word(self, words):

        if self.word_dropout > 0 and self.training:
            mask = torch.rand(words.size())
            mask = mask.to(words.device)
            mask = mask.lt(self.word_dropout)

            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words

    def forward(self, words):

        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words


class BertEmbedding(ContextualEmbedding):
   

    def __init__(self,
                 vocab: Vocabulary,
                 model_dir_or_name: str = 'en-base-uncased',
                 layers: str = '-1',
                 pool_method: str = 'first',
                 word_dropout=0,
                 dropout=0,
                 include_cls_sep: bool = False,
                 pooled_cls=True,
                 requires_grad: bool = True,
                 auto_truncate: bool = False):
       
        super(BertEmbedding, self).__init__(vocab,
                                            word_dropout=word_dropout,
                                            dropout=dropout)
        self.device_cpu = torch.device('cpu')
        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in (
                    'first', 'last'):
                logger.warning(
                    "For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                    " faster speed.")
                warnings.warn(
                    "For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                    " faster speed.")
        self.dropout_p = dropout
        self._word_sep_index = None
        if '[SEP]' in vocab:
            self._word_sep_index = vocab['[SEP]']

        self.model = _WordBertModel(model_dir_or_name=model_dir_or_name,
                                    vocab=vocab,
                                    layers=layers,
                                    pool_method=pool_method,
                                    include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls,
                                    auto_truncate=auto_truncate,
                                    min_freq=2)

        self.requires_grad = requires_grad
        self._embed_size = len(
            self.model.layers) * self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
     
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            if self.dropout_p > 1e-5:
                return self.dropout(outputs)
            else:
                return outputs
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)
        if self.dropout_p > 1e-5:
            return self.dropout(outputs)
        else:
            return outputs

    def drop_word(self, words):
      
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                if self._word_sep_index:  # 不能drop sep
                    sep_mask = words.eq(self._word_sep_index)

                mask = torch.full(words.size(),
                                  fill_value=self.word_dropout,
                                  dtype=torch.float)

                mask = torch.bernoulli(mask).eq(1)
                mask = mask.to(words.device)
                pad_mask = words.ne(0)
                mask = pad_mask.__and__(mask) 
                words = words.masked_fill(mask, self._word_unk_index)
                if self._word_sep_index:
                    words.masked_fill_(sep_mask, self._word_sep_index)
        return words