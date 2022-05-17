import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.LatticeLSTM.bilstmCrf import BiLSTM_CRF
from utils.config import Config
from utils.metric import get_ner_fmeasure

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def config_initialization(config, gaz_file, train_file, dev_file, test_file):
    config.build_alphabet(train_file)
    config.build_alphabet(dev_file)
    config.build_alphabet(test_file)
    config.build_gaz_file(gaz_file)
    config.build_gaz_alphabet(train_file)
    config.build_gaz_alphabet(dev_file)
    config.build_gaz_alphabet(test_file)
    config.fix_alphabet()
    return config


def predict_check(pred_variable, gold_variable, mask_variable):
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlap = (pred == gold)
    right_token = np.sum(overlap * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet,
                  word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.shape[0]
    seq_len = gold_variable.shape[1]
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [
            label_alphabet.get_instance(pred_tag[idx][idy])
            for idy in range(seq_len) if mask[idx][idy] != 0
        ]
        gold = [
            label_alphabet.get_instance(gold_tag[idx][idy])
            for idy in range(seq_len) if mask[idx][idy] != 0
        ]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def save_config_setting(config, save_file):
    new_config = copy.deepcopy(config)
    new_config.train_texts = []
    new_config.dev_texts = []
    new_config.test_texts = []
    new_config.raw_texts = []
    new_config.train_Ids = []
    new_config.dev_Ids = []
    new_config.test_Ids = []
    new_config.raw_Ids = []
    with open(save_file, 'wb') as fp:
        pickle.dump(new_config, fp)
    print("config setting saved to file: ", save_file)


def load_config_setting(save_file):
    with open(save_file, 'rb') as fp:
        config = pickle.load(fp)
    print("config setting loaded from file: ", save_file)
    config.show_config_summary()
    return config


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(config, model, name):
    if name == "train":
        instances = config.train_Ids
    elif name == "dev":
        instances = config.dev_Ids
    elif name == 'test':
        instances = config.test_Ids
    elif name == 'raw':
        instances = config.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
            instance, config.use_cuda, True)
        tag_seq = model(gaz_list, batch_word, batch_biword, batch_wordlen,
                        batch_char, batch_charlen, batch_charrecover, mask)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask,
                                               config.label_alphabet,
                                               batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results,
                                    config.tagScheme)
    return speed, acc, p, r, f, pred_results


def batchify_with_label(input_batch_list, use_cuda, volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    biword_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    mask = torch.zeros((batch_size, max_seq_len)).bool()
    for idx, (seq, biseq, label, seqlen) in enumerate(
            zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.ones(seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    pad_chars = [
        chars[idx] + np.zeros(max_seq_len - len(chars[idx]))
        if max_seq_len > len(chars[idx]) else chars[idx]
        for idx in range(len(chars))
    ]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros(
        (batch_size, max_seq_len, max_word_len)).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(
        batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(
        batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if use_cuda:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(config, save_model_dir, seg=True):
    print("Train model...")
    config.show_config_summary()
    save_config_name = save_model_dir + '.cfg'
    save_config_setting(config, save_config_name)
    model = BiLSTM_CRF(config)
    print("finish build model.")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=config.lr, momentum=config.momentum)
    best_dev = -1
    config.iteration = 100
    for idx in range(config.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.iteration))
        optimizer = lr_decay(optimizer, idx, config.lr_decay, config.lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(config.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = 1
        batch_id = 0
        train_num = len(config.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = config.train_Ids[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, config.use_cuda)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(
                gaz_list, batch_word, batch_biword, batch_wordlen, batch_char,
                batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(
                    "     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"
                    % (end, temp_cost, sample_loss, right_token, whole_token,
                       (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
            if end % config.batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
              (end, temp_cost, sample_loss, right_token, whole_token,
               (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(
            "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"
            % (idx, epoch_cost, train_num / epoch_cost, total_loss))
        speed, acc, p, r, f, _ = evaluate(config, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if seg:
            current_score = f
            print(
                "Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"
                % (dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" %
                  (dev_cost, speed, acc))

        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        speed, acc, p, r, f, _ = evaluate(config, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            print(
                "Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"
                % (test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" %
                  (test_cost, speed, acc))
        gc.collect()


def load_model_decode(model_dir, config, name, use_cuda, seg=True):
    config.use_cuda = use_cuda
    print("Load Model from file: ", model_dir)
    model = BiLSTM_CRF(config)
    model.load_state_dict(torch.load(model_dir))
    print("Decode %s data ..." % (name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(config, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print(
            "%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"
            % (name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" %
              (name, time_cost, speed, acc))
    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding',
                        help='Embedding for words',
                        default='None')
    parser.add_argument('--status',
                        choices=['train', 'test', 'decode'],
                        help='update algorithm',
                        default='train')
    parser.add_argument('--savemodel', default="results/latticeLSTM/")
    parser.add_argument('--savedcfg',
                        help='Dir of saved data setting',
                        default="data/save.cfg")
    parser.add_argument('--train', default="data/demo/demo.train.char")
    parser.add_argument('--dev', default="data/demo/demo.dev.char")
    parser.add_argument('--test', default="data/demo/demo.test.char")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    cfg_dir = args.savedcfg
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    use_cuda = torch.cuda.is_available()

    char_emb = "embeddings/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = None
    gaz_file = "embeddings/ctb.50d.vec"

    print("CuDNN:", torch.backends.cudnn.enabled)
    print("use_cuda available:", use_cuda)
    print("Status:", status)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:", gaz_file)
    if status == 'train':
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        config = Config()
        config.use_cuda = use_cuda
        config.use_char = False
        config.batch_size = 1
        config.use_bigram = False
        config.gaz_dropout = 0.5
        config.norm_gaz_embedding = False
        config.fix_gaz_embedding = False
        config_initialization(config, gaz_file, train_file, dev_file,
                              test_file)
        config.generate_instance_with_gaz(train_file, 'train')
        config.generate_instance_with_gaz(dev_file, 'dev')
        config.generate_instance_with_gaz(test_file, 'test')
        config.build_word_pretrain_embedding(char_emb)
        config.build_biword_pretrain_embedding(bichar_emb)
        config.build_gaz_pretrain_embedding(gaz_file)
        train(config, save_model_dir, seg)
    elif status == 'test':
        config = load_config_setting(cfg_dir)
        config.generate_instance_with_gaz(dev_file, 'dev')
        load_model_decode(model_dir, config, 'dev', use_cuda, seg)
        config.generate_instance_with_gaz(test_file, 'test')
        load_model_decode(model_dir, config, 'test', use_cuda, seg)
    elif status == 'decode':
        config = load_config_setting(cfg_dir)
        config.generate_instance_with_gaz(raw_file, 'raw')
        decode_results = load_model_decode(model_dir, config, 'raw', use_cuda,
                                           seg)
        config.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print(
            "Invalid argument! Please use valid arguments! (train/test/decode)"
        )
