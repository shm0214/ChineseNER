import torch.nn as nn
from model.LatticeLSTM.bilstm import BiLSTM
from model.LatticeLSTM.crf import CRF


class BiLSTM_CRF(nn.Module):

    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.use_cuda = config.use_cuda
        label_size = config.label_alphabet_size
        config.label_alphabet_size += 2
        self.lstm = BiLSTM(config)
        self.crf = CRF(label_size, self.use_cuda)

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs,
                                word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover,
                                batch_label, mask):
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs,
                                          word_seq_lengths, char_inputs,
                                          char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        return total_loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,
                char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs,
                                          word_seq_lengths, char_inputs,
                                          char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        return tag_seq

    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs,
                          word_seq_lengths, char_inputs, char_seq_lengths,
                          char_seq_recover):
        return self.lstm.get_lstm_features(gaz_list, word_inputs,
                                           biword_inputs, word_seq_lengths,
                                           char_inputs, char_seq_lengths,
                                           char_seq_recover)
