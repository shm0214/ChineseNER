import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.LatticeLSTM.LatticeLSTM import LatticeLSTM

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.use_bigram = config.use_bigram
        self.use_cuda = config.use_cuda
        self.batch_size = config.batch_size
        self.char_hidden_dim = 0
        self.biword_embedding_dim = config.biword_embedding_dim
        self.embedding_dim = config.word_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_lstm = nn.Dropout(config.dropout)
        self.word_embeddings = nn.Embedding(config.word_alphabet.size(), self.embedding_dim)
        self.biword_embeddings = nn.Embedding(config.biword_alphabet.size(), self.biword_embedding_dim)
        self.lstm_layer = config.lstm_layer
        self.use_bilstm = config.use_bilstm

        if self.use_bilstm:
            lstm_hidden_dim = self.hidden_dim // 2
        else:
            lstm_hidden_dim = self.hidden_dim

        lstm_input_dim = self.embedding_dim + self.char_hidden_dim
        if self.use_bigram:
            lstm_input_dim += self.biword_embedding_dim
        self.forward_lstm = LatticeLSTM(lstm_input_dim, lstm_hidden_dim, config.gaz_dropout, config.gaz_alphabet.size(), config.gaz_embedding_dim, config.pretrain_gaz_embedding, True, config.fix_gaz_embedding, self.use_cuda)
        if self.use_bilstm:
            self.backward_lstm = LatticeLSTM(lstm_input_dim, lstm_hidden_dim, config.gaz_dropout, config.gaz_alphabet.size(), config.gaz_embedding_dim, config.pretrain_gaz_embedding, False, config.fix_gaz_embedding, self.use_cuda)
        self.hidden2tag = nn.Linear(self.hidden_dim, config.label_alphabet_size)

        if self.use_cuda:
            self.dropout = self.dropout.cuda()
            self.dropout_lstm = self.dropout_lstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.biword_embeddings = self.biword_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.use_bilstm:
                self.backward_lstm = self.backward_lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

    def random_embedding(self, alphabet_size, embedding_dim):
        pretrain_embedding = np.empty([alphabet_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for idx in range(alphabet_size):
            pretrain_embedding[idx, :] = np.ramdom.uniform(
                -scale, scale, [1, embedding_dim])
        return pretrain_embedding

    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        batch_size = word_inputs.shape[0]
        sentence_len = word_inputs.shape[1]
        word_embeddings = self.word_embeddings(word_inputs)
        if self.use_bigram:
            biword_embeddings = self.biword_embeddings(biword_inputs)
            word_embeddings = torch.cat([word_embeddings, biword_embeddings], 2)
        word_embeddings = self.dropout(word_embeddings)
        hidden = None
        lstm_out, hidden = self.forward_lstm(word_embeddings, gaz_list, hidden)
        if self.use_bilstm:
            backward_hidden = None
            backward_lstm_out, backward_hidden = self.forward_lstm(word_embeddings, gaz_list, backward_hidden)
            lstm_out = torch.cat([lstm_out, backward_lstm_out], 2)
        lstm_out = self.dropout_lstm(lstm_out)
        return lstm_out

    def get_output_score(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        lstm_out = self.get_lstm_features(gaz_list, word_inputs, biword_inputs,
                                          word_seq_lengths, char_inputs,
                                          char_seq_lengths, char_seq_recover)
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs,
                                word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover,
                                batch_label):
        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        total_word = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = self.get_output_score(gaz_list, word_inputs, biword_inputs,
                                     word_seq_lengths, char_inputs,
                                     char_seq_lengths, char_seq_recover)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,
                char_inputs, char_seq_lengths, char_seq_recover, mask):

        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        total_word = batch_size * seq_len
        outs = self.get_output_score(gaz_list, word_inputs, biword_inputs,
                                     word_seq_lengths, char_inputs,
                                     char_seq_lengths, char_seq_recover)
        outs = outs.view(total_word, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        decode_seq = mask.long() * tag_seq
        return decode_seq
