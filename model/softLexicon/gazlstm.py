import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.softLexicon.crf import CRF
from model.softLexicon.layers import NERmodel
from transformers import BertModel


class GazLSTM(nn.Module):

    def __init__(self, config):
        super(GazLSTM, self).__init__()

        self.use_biword = config.use_bigram
        self.use_cuda = config.use_cuda
        self.hidden_dim = config.hidden_dim
        self.gaz_alphabet = config.gaz_alphabet
        self.gaz_embedding_dim = config.gaz_embedding_dim
        self.word_embedding_dim = config.word_embedding_dim
        self.biword_embedding_dim = config.biword_embedding_dim
        self.use_char = config.use_char
        self.bilstm_flag = config.use_bilstm
        self.lstm_layer = config.lstm_layer
        self.use_count = config.use_count
        self.num_layer = config.num_layer
        self.model_type = config.model_type
        self.use_bert = config.use_bert

        scale = np.sqrt(3.0 / self.gaz_embedding_dim)
        config.pretrain_gaz_embedding[0, :] = np.random.uniform(
            -scale, scale, [1, self.gaz_embedding_dim])

        if self.use_char:
            scale = np.sqrt(3.0 / self.word_embedding_dim)
            config.pretrain_word_embedding[0, :] = np.random.uniform(
                -scale, scale, [1, self.word_embedding_dim])

        self.gaz_embedding = nn.Embedding(config.gaz_alphabet.size(),
                                          self.gaz_embedding_dim)
        self.word_embedding = nn.Embedding(config.word_alphabet.size(),
                                           self.word_embedding_dim)
        if self.use_biword:
            self.biword_embedding = nn.Embedding(config.biword_alphabet.size(),
                                                 self.biword_embedding_dim)

        if config.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(
                torch.from_numpy(config.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(config.gaz_alphabet.size(),
                                          self.gaz_embedding_dim)))

        if config.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(config.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(config.word_alphabet.size(),
                                          self.word_embedding_dim)))
        if self.use_biword:
            if config.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(
                    torch.from_numpy(config.pretrain_biword_embedding))
            else:
                self.biword_embedding.weight.data.copy_(
                    torch.from_numpy(
                        self.random_embedding(config.biword_alphabet.size(),
                                              self.word_embedding_dim)))

        char_feature_dim = self.word_embedding_dim + 4 * self.gaz_embedding_dim
        if self.use_biword:
            char_feature_dim += self.biword_embedding_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768

        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            self.NERmodel = NERmodel(model_type='lstm',
                                     input_dim=char_feature_dim,
                                     hidden_dim=lstm_hidden,
                                     num_layer=self.lstm_layer,
                                     biflag=self.bilstm_flag)

        if self.model_type == 'cnn':
            self.NERmodel = NERmodel(model_type='cnn',
                                     input_dim=char_feature_dim,
                                     hidden_dim=self.hidden_dim,
                                     num_layer=self.num_layer,
                                     dropout=config.dropout,
                                     use_cuda=self.use_cuda)

        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer',
                                     input_dim=char_feature_dim,
                                     hidden_dim=self.hidden_dim,
                                     num_layer=self.num_layer,
                                     dropout=config.dropout)

        self.drop = nn.Dropout(p=config.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim,
                                    config.label_alphabet_size + 2)
        self.crf = CRF(config.label_alphabet_size, self.use_cuda)

        if self.use_bert:
            self.bert_encoder = BertModel.from_pretrained('bert-base-chinese')
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        if self.use_cuda:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()
            self.NERmodel = self.NERmodel.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()

    def get_tags(self, gaz_list, word_inputs, biword_inputs, layer_gaz,
                 gaz_count, gaz_chars, gaz_mask_input, gazchar_mask_input,
                 mask, word_seq_lengths, batch_bert, bert_mask):

        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        max_gaz_num = layer_gaz.shape[-1]
        gaz_match = []

        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs], dim=-1)

        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)
        else:
            word_inputs_d = word_embs

        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(
                1, 1, 1, 1, 1, self.word_embedding_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(
                gazchar_mask.data, 0)

            gaz_charnum = (gazchar_mask_input == 0).sum(
                dim=-1, keepdim=True).float()
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        else:
            gaz_embeds = self.gaz_embedding(layer_gaz)

            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
            else:
                gaz_embeds_d = gaz_embeds

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(
                1, 1, 1, 1, self.gaz_embedding_dim)

            gaz_embeds = gaz_embeds_d.data.masked_fill_(
                gaz_mask.data, 0)

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)

            weights = gaz_count.div(count_sum)
            weights = weights * 4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights * gaz_embeds
            gaz_embeds = torch.sum(gaz_embeds, dim=3)

        else:
            gaz_num = (gaz_mask_input == 0).sum(
                dim=-1, keepdim=True).float()
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num

        gaz_embeds_cat = gaz_embeds.view(batch_size, seq_len, -1)

        word_input_cat = torch.cat([word_inputs_d, gaz_embeds_cat],
                                   dim=-1)


        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            outputs = outputs[0][:, 1:-1, :]
            word_input_cat = torch.cat([word_input_cat, outputs], dim=-1)

        feature_out_d = self.NERmodel(word_input_cat)

        tags = self.hidden2tag(feature_out_d)

        return tags, gaz_match

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs,
                                word_seq_lengths, layer_gaz, gaz_count,
                                gaz_chars, gaz_mask, gazchar_mask, mask,
                                batch_label, batch_bert, bert_mask):

        tags, _ = self.get_tags(gaz_list, word_inputs, biword_inputs,
                                layer_gaz, gaz_count, gaz_chars, gaz_mask,
                                gazchar_mask, mask, word_seq_lengths,
                                batch_bert, bert_mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf.viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,
                layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask,
                batch_bert, bert_mask):

        tags, gaz_match = self.get_tags(gaz_list, word_inputs, biword_inputs,
                                        layer_gaz, gaz_count, gaz_chars,
                                        gaz_mask, gazchar_mask, mask,
                                        word_seq_lengths, batch_bert,
                                        bert_mask)

        scores, tag_seq = self.crf.viterbi_decode(tags, mask)

        return tag_seq, gaz_match
