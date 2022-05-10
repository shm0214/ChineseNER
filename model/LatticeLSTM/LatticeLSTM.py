from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class WordLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, use_bias=True):
        super(WordLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_dim, 3 * hidden_dim))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(input_dim, 3 * hidden_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_dim)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, val=0)

    def forward(self, input, hx):
        h_0, c_0 = hx
        batch_size = h_0.shape[0]
        bias_batch = self.bias.unsqueeze(0).expand(batch_size,
                                                   *self.bias.shape)
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input, self.weight_ih)
        f, i, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_dim, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        return c_1

    def __repr__(self):
        s = '{name}({input_dim}, {hidden_dim})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, use_bias=True):
        super(MultiInputLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_dim, 3 * hidden_dim))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(input_dim, 3 * hidden_dim))
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_dim, hidden_dim))
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(input_dim, hidden_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_dim))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_dim))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.alpha_weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_dim)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        alpha_weight_hh_data = torch.eye(self.hidden_dim)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        with torch.no_grad():
            self.alpha_weight_hh.set_(alpha_weight_hh_data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, val=0)
            nn.init.constant_(self.alpha_bias.data, val=0)

    def forward(self, input, c_input, hx):
        h_0, c_0 = hx
        batch_size = h_0.shape[0]
        assert (batch_size == 1)
        bias_batch = self.bias.unsqueeze(0).expand(batch_size,
                                                   *self.bias.shape)
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input, self.weight_ih)
        i, o, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_num = len(c_input)
        if c_num == 0:
            f = 1 - i
            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            c_input_var = torch.cat(c_input, 0)
            alpha_bias_batch = self.alpha_bias.unsqueeze(0).expand(
                batch_size, *self.alpha_bias.shape)
            c_input_var = c_input_var.squeeze(1)
            alpha_wi = torch.addmm(alpha_bias_batch, input,
                                   self.alpha_weight_ih).expand(
                                       c_num, self.hidden_dim)
            alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)
            alpha = torch.sigmoid(alpha_wi + alpha_wh)
            alpha = torch.exp(torch.cat([i, alpha], 0))
            alpha_sum = alpha.sum(0)
            alpha = torch.div(alpha, alpha_sum)
            merge_i_c = torch.cat([g, c_input_var], 0)
            c_1 = merge_i_c * alpha
            c_1 = c_1.sum(0).unsqueeze(0)
            h_1 = o * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_dim}, {hidden_dim})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 word_drop,
                 word_alphabet_size,
                 word_embedding_dim,
                 pretrain_word_embedding=None,
                 left2right=True,
                 fix_word_embedding=True,
                 use_gpu=True,
                 use_bias=True):
        super(LatticeLSTM, self).__init__()
        skip_direction = "forward" if left2right else "backward"
        print("build LatticeLSTM...", skip_direction, ", Fix embedding:",
              fix_word_embedding, " gaz drop:", word_drop)
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(word_alphabet_size,
                                           word_embedding_dim)
        if pretrain_word_embedding is not None:
            print("load pretrain word embedding...",
                  pretrain_word_embedding.shape)
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(word_alphabet_size,
                                          word_embedding_dim)))
        if fix_word_embedding:
            self.word_embedding.weight.requires_grad = False
        self.word_dropout = nn.Dropout(word_drop)
        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim, use_bias)
        self.word_rnn = WordLSTMCell(word_embedding_dim, hidden_dim, use_bias)
        self.left2right = left2right
        if self.use_gpu:
            self.rnn = self.rnn.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.word_dropout = self.word_dropout.cuda()
            self.word_rnn = self.word_rnn.cuda()

    def random_embedding(self, alphabet_size, embedding_dim):
        pretrain_embedding = np.empty([alphabet_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for idx in range(alphabet_size):
            pretrain_embedding[idx, :] = np.ramdom.uniform(
                -scale, scale, [1, embedding_dim])
        return pretrain_embedding

    def forward(self, input, skip_input, hidden=None):
        volatile_flag = skip_input[1]
        skip_input = skip_input[0]
        if not self.left2right:
            skip_input = convert_forward_gaz_to_backward(skip_input)
        input = input.transpose(1, 0)
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        assert (batch_size == 1)
        hidden_out = []
        memory_out = []
        if hidden:
            hx, cx = hidden
        else:
            hx = torch.autograd.Variable(
                torch.zeros(batch_size, self.hidden_dim))
            cx = torch.autograd.Variable(
                torch.zeros(batch_size, self.hidden_dim))
        if self.use_gpu:
            hx = hx.cuda()
            cx = cx.cuda()
        id_list = range(seq_len)
        if not self.left2right:
            id_list = list(reversed(id_list))
        input_char_list = init_list_of_objects(seq_len)
        for t in id_list:
            hx, cx = self.rnn(input[t], input_char_list[t], (hx, cx))
            hidden_out.append(hx)
            memory_out.append(cx)
            if skip_input[t]:
                matched_num = len(skip_input[t][0])
                word_var = torch.autograd.Variable(torch.LongTensor(
                    skip_input[t][0]))
                if self.use_gpu:
                    word_var = word_var.cuda()
                word_embedding = self.word_embedding(word_var)
                word_embedding = self.word_dropout(word_embedding)
                ct = self.word_rnn(word_embedding, (hx, cx))
                assert (ct.shape[0] == len(skip_input[t][1]))
                for idx in range(matched_num):
                    length = skip_input[t][1][idx]
                    if self.left2right:
                        input_char_list[t + length - 1].append(
                            ct[idx, :].unsqueeze(0))
                    else:
                        input_char_list[t - length + 1].append(
                            ct[idx, :].unsqueeze(0))
        if not self.left2right:
            hidden_out = list(reversed(hidden_out))
            memory_out = list(reversed(memory_out))
        output_hidden, output_memory = torch.cat(hidden_out,
                                                 0), torch.cat(memory_out, 0)
        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)

def init_list_of_objects(size):
    list_of_objects = list()
    for _ in range(size):
        list_of_objects.append(list())
    return list_of_objects

def convert_forward_gaz_to_backward(forward_gaz):
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)
    for idx in range(length):
        if forward_gaz[idx]:
            assert(len(forward_gaz[idx]) == 2)
            num = len(forward_gaz[idx][0])
            for i in range(num):
                word_id = forward_gaz[idx][0][i]
                word_length = forward_gaz[idx][1][i]
                new_pos = idx + word_length - 1
                if backward_gaz[new_pos]:
                    backward_gaz[new_pos][0].append(word_id)
                    backward_gaz[new_pos][1].append(word_length)
                else:
                    backward_gaz[new_pos] = [[word_id], [word_length]]
    return backward_gaz
