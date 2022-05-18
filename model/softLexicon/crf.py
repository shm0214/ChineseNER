import torch
import torch.autograd as autograd
import torch.nn as nn

START_TAG = -2
STOP_TAG = -1


def log_sum_exp(vec, hidden_dim):
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1,
                             idx.view(-1, 1,
                                      hidden_dim)).view(-1, 1, hidden_dim)
    return max_score.view(-1, hidden_dim) + torch.log(
        torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(
            -1, hidden_dim)


class CRF(nn.Module):

    def __init__(self, tagset_size, use_cuda):
        super(CRF, self).__init__()
        self.use_cuda = use_cuda
        self.average_batch = False
        self.tagset_size = tagset_size
        init_transitions = torch.zeros(self.tagset_size + 2,
                                       self.tagset_size + 2)
        if self.use_cuda:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

    def calculate_P(self, feats, mask):
        batch_size = feats.shape[0]
        seq_len = feats.shape[1]
        tag_size = feats.shape[2]
        assert (tag_size == self.tagset_size + 2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, init_values = next(seq_iter)
        partition = init_values[:, START_TAG, :].clone().view(
            batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            mask_idx = mask[idx, :].view(batch_size,
                                         1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def viterbi_decode(self, feats, mask):
        batch_size = feats.shape[0]
        seq_len = feats.shape[1]
        tag_size = feats.shape[2]
        assert (tag_size == self.tagset_size + 2)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(
            ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).bool()
        _, inivalues = next(seq_iter)
        partition = inivalues[:, START_TAG, :].clone().view(
            batch_size, tag_size, 1)
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(2))
            cur_bp.masked_fill_(
                mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history,
                                      0).view(seq_len, batch_size,
                                              -1).transpose(1, 0).contiguous()
        last_position = length_mask.view(batch_size, 1, 1).expand(
            batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1,
                                      last_position).view(
                                          batch_size, tag_size, 1)
        last_values = last_partition.expand(
            batch_size, tag_size, tag_size) + self.transitions.view(
                1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.use_cuda:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size,
                                                  tag_size)

        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(
            batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1,
                                   pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.squeeze(1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats):
        path_score, best_path = self.viterbi_decode(feats)
        return path_score, best_path

    def score_sentence(self, scores, mask, tags):
        batch_size = scores.shape[1]
        seq_len = scores.shape[0]
        tag_size = scores.shape[2]
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(
            seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2,
                                 new_tags).view(
                                     seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        batch_size = feats.shape[0]
        forward_score, scores = self.calculate_P(feats, mask)
        gold_score = self.score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        else:
            return forward_score - gold_score
