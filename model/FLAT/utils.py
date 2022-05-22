import collections

def get_skip_path(chars, w_trie):
    sentence = ''.join(chars)
    result = w_trie.get_lexicon(sentence)

    return result


def get_skip_path_trivial(chars, w_list):
    chars = ''.join(chars)
    w_set = set(w_list)
    result = []
    for i in range(len(chars) - 1):
        for j in range(i + 2, len(chars) + 1):
            if chars[i:j] in w_set:
                result.append([i, j - 1, chars[i:j]])

    return result


class TrieNode:

    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self, w):
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i, j, sentence[i:j + 1]])

        return result


from fastNLP.core.field import Padder
import numpy as np
import torch
from collections import defaultdict


class LatticeLexiconPadder(Padder):

    def __init__(self,
                 pad_val=0,
                 pad_val_dynamic=False,
                 dynamic_offset=0,
                 **kwargs):
        '''

        :param pad_val:
        :param pad_val_dynamic: if True, pad_val is the seq_len
        :param kwargs:
        '''
        self.pad_val = pad_val
        self.pad_val_dynamic = pad_val_dynamic
        self.dynamic_offset = dynamic_offset

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_len = max(map(len, contents))

        max_len = max(max_len, 1)

        max_word_len = max([
            max([len(content_ii) for content_ii in content_i])
            for content_i in contents
        ])

        max_word_len = max(max_word_len, 1)
        if self.pad_val_dynamic:
            array = np.full((len(contents), max_len, max_word_len),
                            max_len - 1 + self.dynamic_offset,
                            dtype=field_ele_dtype)

        else:
            array = np.full((len(contents), max_len, max_word_len),
                            self.pad_val,
                            dtype=field_ele_dtype)
        for i, content_i in enumerate(contents):
            for j, content_ii in enumerate(content_i):
                array[i, j, :len(content_ii)] = content_ii
        array = torch.tensor(array)

        return array


from fastNLP.core.metrics import MetricBase


def get_yangjie_bmeso(label_list, ignore_labels=None):

    def get_ner_BMESO_yj(label_list):

        def reverse_style(input_string):
            target_position = input_string.index('[')
            input_len = len(input_string)
            output_string = input_string[
                target_position:input_len] + input_string[0:target_position]
            return output_string

        list_len = len(label_list)
        begin_label = 'b-'
        end_label = 'e-'
        single_label = 's-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            current_label = label_list[i].lower()
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "",
                                                  1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(single_label, "",
                                                  1) + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        return stand_matrix

    def transform_YJ_to_fastNLP(span):
        span = span[1:]
        span_split = span.split(']')
        span_type = span_split[1]
        if ',' in span_split[0]:
            b, e = span_split[0].split(',')
        else:
            b = span_split[0]
            e = b

        b = int(b)
        e = int(e)

        e += 1

        return (span_type, (b, e))

    yj_form = get_ner_BMESO_yj(label_list)
    fastNLP_form = list(map(transform_YJ_to_fastNLP, yj_form))
    return fastNLP_form


class SpanFPreRecMetric_YJ(MetricBase):

    def __init__(self,
                 tag_vocab,
                 pred=None,
                 target=None,
                 seq_len=None,
                 encoding_type='bio',
                 ignore_labels=None,
                 only_gross=True,
                 f_type='micro',
                 beta=1):
        from fastNLP.core import Vocabulary
        from fastNLP.core.metrics import _bmes_tag_to_spans,_bio_tag_to_spans,\
            _bioes_tag_to_spans,_bmeso_tag_to_spans
        from collections import defaultdict

        encoding_type = encoding_type.lower()

        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError(
                "tag_vocab can only be fastNLP.Vocabulary, not {}.".format(
                    type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError(
                "f_type only supports `micro` or `macro`', got {}.".format(
                    f_type))

        self.encoding_type = encoding_type
        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        elif self.encoding_type == 'bmesoyj':
            self.tag_to_span_func = get_yangjie_bmeso
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta**2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len):
        from fastNLP.core.utils import _get_func_signature

        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(
                f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(
                target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError(
                    "A gold label passed to SpanBasedF1Metric contains an "
                    "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(
                f"In {_get_func_signature(self.evaluate)}, when pred have "
                f"size:{pred.size()}, target should have size: {pred.size()} or "
                f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(
                pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(
                gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = self._compute_f_pre_rec(
                sum(self._true_positives.values()),
                sum(self._false_negatives.values()),
                sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):

        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre +
                                                  rec + 1e-13)

        return f, pre, rec


import torch
import random
import numpy as np
import os
import torch.nn as nn
from fastNLP import logger


class MyDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.001:
            mask = torch.rand(x.size())
            mask = mask.to(x)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0) / (1 - self.p)
        return x


def should_mask(name, t=''):
    if 'bias' in name:
        return False
    if 'embedding' in name:
        splited = name.split('.')
        if splited[-1] != 'weight':
            return False
        if 'embedding' in splited[-2]:
            return False
    if 'c0' in name:
        return False
    if 'h0' in name:
        return False

    if 'output' in name and t not in name:
        return False

    return True


def get_init_mask(model):
    init_masks = {}
    for name, param in model.named_parameters():
        if should_mask(name):
            init_masks[name + '.mask'] = torch.ones_like(param)

    return init_masks


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 100)
    torch.manual_seed(seed + 200)
    torch.cuda.manual_seed_all(seed + 300)


def get_parameters_size(model):
    result = {}
    for name, p in model.state_dict().items():
        result[name] = p.size()

    return result


def prune_by_proportion_model(model, proportion, task):
    for name, p in model.named_parameters():
        if not should_mask(name, task):
            continue

        tensor = p.data.cpu().numpy()
        index = np.nonzero(model.mask[task][name + '.mask'].data.cpu().numpy())
        alive = tensor[index]
        percentile_value = np.percentile(abs(alive), (1 - proportion) * 100)
        prune_by_threshold_parameter(p, model.mask[task][name + '.mask'],
                                     percentile_value)


def prune_by_proportion_model_global(model, proportion, task):
    alive = None
    for name, p in model.named_parameters():
        if not should_mask(name, task):
            continue

        tensor = p.data.cpu().numpy()
        index = np.nonzero(model.mask[task][name + '.mask'].data.cpu().numpy())
        if alive is None:
            alive = tensor[index]
        else:
            alive = np.concatenate([alive, tensor[index]], axis=0)

    percentile_value = np.percentile(abs(alive), (1 - proportion) * 100)

    for name, p in model.named_parameters():
        if should_mask(name, task):
            prune_by_threshold_parameter(p, model.mask[task][name + '.mask'],
                                         percentile_value)


def prune_by_threshold_parameter(p, mask, threshold):
    p_abs = torch.abs(p)

    new_mask = (p_abs > threshold).float()
    mask[:] *= new_mask


def one_time_train_and_prune_single_task(
    trainer,
    PRUNE_PER,
    optimizer_init_state_dict=None,
    model_init_state_dict=None,
    is_global=None,
):


    trainer.optimizer.load_state_dict(optimizer_init_state_dict)
    trainer.model.load_state_dict(model_init_state_dict)
    trainer.train(load_best_model=True)
    if is_global:

        prune_by_proportion_model_global(trainer.model, PRUNE_PER,
                                         trainer.model.now_task)

    else:
        prune_by_proportion_model(trainer.model, PRUNE_PER,
                                  trainer.model.now_task)


def iterative_train_and_prune_single_task(get_trainer,
                                          args,
                                          model,
                                          train_set,
                                          dev_set,
                                          test_set,
                                          device,
                                          save_path=None):
    import torch
    import math
    import copy
    PRUNE = args.prune
    ITER = args.iter
    trainer = get_trainer(args, model, train_set, dev_set, test_set, device)
    optimizer_init_state_dict = copy.deepcopy(trainer.optimizer.state_dict())
    model_init_state_dict = copy.deepcopy(trainer.model.state_dict())
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    mask_count = 0
    model = trainer.model
    task = trainer.model.now_task
    for name, p in model.mask[task].items():
        mask_count += torch.sum(p).item()
    init_mask_count = mask_count
    logger.info('init mask count:{}'.format(mask_count))


    prune_per_iter = math.pow(PRUNE, 1 / ITER)

    for i in range(ITER):
        trainer = get_trainer(args, model, train_set, dev_set, test_set,
                              device)
        one_time_train_and_prune_single_task(trainer, prune_per_iter,
                                             optimizer_init_state_dict,
                                             model_init_state_dict)
        if save_path is not None:
            f = open(
                os.path.join(save_path, task + '_mask_' + str(i) + '.pkl'),
                'wb')
            torch.save(model.mask[task], f)

        mask_count = 0
        for name, p in model.mask[task].items():
            mask_count += torch.sum(p).item()
        logger.info('{}th traning mask count: {} / {} = {}%'.format(
            i, mask_count, init_mask_count,
            mask_count / init_mask_count * 100))


def get_appropriate_cuda(task_scale='s'):
    if task_scale not in {'s', 'm', 'l'}:
        logger.info('task scale wrong!')
        exit(2)
    import pynvml
    pynvml.nvmlInit()
    total_cuda_num = pynvml.nvmlDeviceGetCount()
    for i in range(total_cuda_num):
        logger.info(i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 这里的0是GPU id
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilizationInfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        logger.info(i, 'mem:', memInfo.used / memInfo.total, 'util:',
                    utilizationInfo.gpu)
        if memInfo.used / memInfo.total < 0.15 and utilizationInfo.gpu < 0.2:
            logger.info(i, memInfo.used / memInfo.total)
            return 'cuda:' + str(i)

    if task_scale == 's':
        max_memory = 2000
    elif task_scale == 'm':
        max_memory = 6000
    else:
        max_memory = 9000

    max_id = -1
    for i in range(total_cuda_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
        memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilizationInfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if max_memory < memInfo.free:
            max_memory = memInfo.free
            max_id = i

    if id == -1:
        logger.info('no appropriate gpu, wait!')
        exit(2)

    return 'cuda:' + str(max_id)

    # if memInfo.used / memInfo.total < 0.5:
    #     return


def print_mask(mask_dict):

    def seq_mul(*X):
        res = 1
        for x in X:
            res *= x
        return res

    for name, p in mask_dict.items():
        total_size = seq_mul(*p.size())
        unmasked_size = len(np.nonzero(p))

        print(name, ':', unmasked_size, '/', total_size, '=',
              unmasked_size / total_size * 100, '%')

    print()


def check_words_same(dataset_1, dataset_2, field_1, field_2):
    if len(dataset_1[field_1]) != len(dataset_2[field_2]):
        logger.info('CHECK: example num not same!')
        return False

    for i, words in enumerate(dataset_1[field_1]):
        if len(dataset_1[field_1][i]) != len(dataset_2[field_2][i]):
            logger.info('CHECK {} th example length not same'.format(i))
            logger.info('1:{}'.format(dataset_1[field_1][i]))
            logger.info('2:'.format(dataset_2[field_2][i]))
            return False

        # for j,w in enumerate(words):
        #     if dataset_1[field_1][i][j] != dataset_2[field_2][i][j]:
        #         print('CHECK', i, 'th example has words different!')
        #         print('1:',dataset_1[field_1][i])
        #         print('2:',dataset_2[field_2][i])
        #         return False

    logger.info('CHECK: totally same!')

    return True


def get_now_time():
    import time
    from datetime import datetime, timezone, timedelta
    dt = datetime.utcnow()
    # print(dt)
    tzutc_8 = timezone(timedelta(hours=8))
    local_dt = dt.astimezone(tzutc_8)
    result = ("_{}_{}_{}__{}_{}_{}".format(local_dt.year, local_dt.month,
                                           local_dt.day, local_dt.hour,
                                           local_dt.minute, local_dt.second))

    return result


def get_bigrams(words):
    result = []
    for i, w in enumerate(words):
        if i != len(words) - 1:
            result.append(words[i] + words[i + 1])
        else:
            result.append(words[i] + '<end>')

    return result


def print_info(*inp, islog=True, sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp, sep=sep)
    else:
        inp = sep.join(map(str, inp))
        logger.info(inp)


def better_init_rnn(rnn, coupled=False):
    import torch.nn as nn
    if coupled:
        repeat_size = 3
    else:
        repeat_size = 4
    # print(list(rnn.named_parameters()))
    if hasattr(rnn, 'num_layers'):
        for i in range(rnn.num_layers):
            nn.init.orthogonal_(getattr(rnn, 'weight_ih_l' + str(i)).data)
            weight_hh_data = torch.eye(rnn.hidden_size)
            weight_hh_data = weight_hh_data.repeat(1, repeat_size)
            with torch.no_grad():
                getattr(rnn, 'weight_hh_l' + str(i)).set_(weight_hh_data)
            nn.init.constant_(getattr(rnn, 'bias_ih_l' + str(i)).data, val=0)
            nn.init.constant_(getattr(rnn, 'bias_hh_l' + str(i)).data, val=0)

        if rnn.bidirectional:
            for i in range(rnn.num_layers):
                nn.init.orthogonal_(
                    getattr(rnn, 'weight_ih_l' + str(i) + '_reverse').data)
                weight_hh_data = torch.eye(rnn.hidden_size)
                weight_hh_data = weight_hh_data.repeat(1, repeat_size)
                with torch.no_grad():
                    getattr(rnn, 'weight_hh_l' + str(i) +
                            '_reverse').set_(weight_hh_data)
                nn.init.constant_(getattr(rnn, 'bias_ih_l' + str(i) +
                                          '_reverse').data,
                                  val=0)
                nn.init.constant_(getattr(rnn, 'bias_hh_l' + str(i) +
                                          '_reverse').data,
                                  val=0)

    else:
        nn.init.orthogonal_(rnn.weight_ih.data)
        weight_hh_data = torch.eye(rnn.hidden_size)
        weight_hh_data = weight_hh_data.repeat(repeat_size, 1)
        with torch.no_grad():
            rnn.weight_hh.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        print('rnn param size:{},{}'.format(rnn.weight_hh.size(), type(rnn)))
        if rnn.bias:
            nn.init.constant_(rnn.bias_ih.data, val=0)
            nn.init.constant_(rnn.bias_hh.data, val=0)

    # print(list(rnn.named_parameters()))


def get_crf_zero_init(label_size,
                      include_start_end_trans=False,
                      allowed_transitions=None,
                      initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(
        torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(
            torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(
            torch.zeros(size=[label_size], requires_grad=True))
    return crf


def get_peking_time():
    import time
    import datetime
    import pytz

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(
        int(time.time()),
        pytz.timezone('Asia/Shanghai')).strftime('%Y_%m_%d_%H_%M_%S')
    return t


def norm_static_embedding(x, norm=1):
    with torch.no_grad():
        x.embedding.weight /= (
            torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(),
                                             para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print(
        'Model {} : intermedite variables: {:3f} M (without backward)'.format(
            model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'.format(
        model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def size2MB(size_, type_size=4):
    num = 1
    for s in size_:
        num *= s

    return num * type_size / 1000 / 1000


if __name__ == '__main__':
    a = get_peking_time()
    print(a)
    print(type(a))
