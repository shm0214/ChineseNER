import sys
from utils.alphabet import Alphabet
from utils.functions import *
from utils.gazetteer import Gazetteer

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Config:

    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_embedding = True
        self.norm_biword_embedding = True
        self.norm_gaz_embedding = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')

        self.gaz_count = {}
        self.gaz_split = {}
        self.biword_count = {}

        self.fix_gaz_embedding = False
        self.use_gaz = True
        self.use_count = False

        self.tagScheme = "NoSeg"
        self.char_features = "LSTM"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.train_split_index = []
        self.dev_split_index = []

        self.use_bigram = True
        self.word_embedding_dim = 50
        self.biword_embedding_dim = 50
        self.char_embedding_dim = 30
        self.gaz_embedding_dim = 50
        self.gaz_dropout = 0.5
        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.iteration = 100
        self.batch_size = 10
        self.char_hidden_dim = 50
        self.hidden_dim = 200
        self.dropout = 0.5
        self.lstm_layer = 1
        self.use_bilstm = True
        self.use_char = False
        self.use_cuda = False
        self.lr = 0.015
        self.lr_decay = 0.05
        self.clip = 5.0
        self.momentum = 0
        self.num_layer = 4

    def show_config_summary(self):
        print("CONFIG SUMMARY START:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bigram))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Biword alphabet size: %s" % (self.biword_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Gaz   alphabet size: %s" % (self.gaz_alphabet.size()))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Word embedding size: %s" % (self.word_embedding_dim))
        print("     Biword embedding size: %s" % (self.biword_embedding_dim))
        print("     Char embedding size: %s" % (self.char_embedding_dim))
        print("     Gaz embedding size: %s" % (self.gaz_embedding_dim))
        print("     Norm     word   embedding: %s" %
              (self.norm_word_embedding))
        print("     Norm     biword embedding: %s" %
              (self.norm_biword_embedding))
        print("     Norm     gaz    embedding: %s" % (self.norm_gaz_embedding))
        print("     Norm   gaz  dropout: %s" % (self.gaz_dropout))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Hyperpara  iteration: %s" % (self.iteration))
        print("     Hyperpara  batch size: %s" % (self.batch_size))
        print("     Hyperpara          lr: %s" % (self.lr))
        print("     Hyperpara    lr_decay: %s" % (self.lr_decay))
        print("     Hyperpara     clip: %s" % (self.clip))
        print("     Hyperpara    momentum: %s" % (self.momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.use_bilstm))
        print("     Hyperpara         GPU: %s" % (self.use_cuda))
        print("     Hyperpara     use_gaz: %s" % (self.use_gaz))
        print("     Hyperpara fix gaz embedding: %s" %
              (self.fix_gaz_embedding))
        print("     Hyperpara    use_char: %s" % (self.use_char))
        if self.use_char:
            print("             Char_features: %s" % (self.char_features))
        print("CONFIG SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.items():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s" %
              (old_size, self.label_alphabet_size))

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        seqlen = 0
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    biword = word + in_lines[idx + 1].strip().split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                for char in word:
                    self.char_alphabet.add(char)

                seqlen += 1
            else:
                seqlen = 0
        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.items():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_gaz_file(self, gaz_file):
        if gaz_file:
            fins = open(gaz_file, 'r', encoding='utf-8').readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    self.gaz.insert(fin, "one_source")
            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            print("Gaz file is None, load nothing")

    def build_gaz_alphabet(self, input_file, count=False):
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 3:
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                w_length = len(word_list)
                entitys = []
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(
                        word_list[idx:])
                    for entity in matched_entity:
                        self.gaz_alphabet.add(entity)
                        index = self.gaz_alphabet.get_index(entity)
                        self.gaz_count[index] = self.gaz_count.get(index, 0)
                if count:
                    entitys.sort(key=lambda x: -len(x))
                    while entitys:
                        longest = entitys[0]
                        longest_index = self.gaz_alphabet.get_index(longest)
                        self.gaz_count[longest_index] = self.gaz_count.get(
                            longest_index, 0) + 1

                        gazlen = len(longest)
                        for i in range(gazlen):
                            for j in range(i + 1, gazlen + 1):
                                covering_gaz = longest[i:j]
                                if covering_gaz in entitys:
                                    entitys.remove(covering_gaz)
                word_list = []
        print("gaz alphabet size:", self.gaz_alphabet.size())

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_word_pretrain_embedding(self, embedding_path):
        print("build word pretrain embedding...")
        self.pretrain_word_embedding, self.word_embedding_dim = build_pretrain_embedding(
            embedding_path, self.word_alphabet, self.word_embedding_dim,
            self.norm_word_embedding)

    def build_biword_pretrain_embedding(self, embedding_path):
        print("build biword pretrain embedding...")
        self.pretrain_biword_embedding, self.biword_embedding_dim = build_pretrain_embedding(
            embedding_path, self.biword_alphabet, self.biword_embedding_dim,
            self.norm_biword_embedding)

    def build_gaz_pretrain_embedding(self, embedding_path):
        print("build gaz pretrain embedding...")
        self.pretrain_gaz_embedding, self.gaz_embedding_dim = build_pretrain_embedding(
            embedding_path, self.gaz_alphabet, self.gaz_embedding_dim,
            self.norm_gaz_embedding)

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(
                input_file, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(
                input_file, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(
                input_file, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(
                input_file, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print(
                "Error: you can only generate train/dev/test instance! Illegal input:%s"
                % (name))

    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(
                input_file, self.gaz, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.gaz_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(
                input_file, self.gaz, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.gaz_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(
                input_file, self.gaz, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.gaz_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(
                input_file, self.gaz, self.word_alphabet, self.biword_alphabet,
                self.char_alphabet, self.gaz_alphabet, self.label_alphabet,
                self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print(
                "Error: you can only generate train/dev/test instance! Illegal input:%s"
                % (name))

    def generate_instance_with_gaz_softlexicon(self, input_file, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz_softLexicon(
                self.num_layer, input_file, self.gaz, self.word_alphabet,
                self.biword_alphabet, self.biword_count, self.char_alphabet,
                self.gaz_alphabet, self.gaz_count, self.gaz_split,
                self.label_alphabet, self.number_normalized,
                self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz_softLexicon(
                self.num_layer, input_file, self.gaz, self.word_alphabet,
                self.biword_alphabet, self.biword_count, self.char_alphabet,
                self.gaz_alphabet, self.gaz_count, self.gaz_split,
                self.label_alphabet, self.number_normalized,
                self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz_softLexicon(
                self.num_layer, input_file, self.gaz, self.word_alphabet,
                self.biword_alphabet, self.biword_count, self.char_alphabet,
                self.gaz_alphabet, self.gaz_count, self.gaz_split,
                self.label_alphabet, self.number_normalized,
                self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz_softLexicon(
                self.num_layer, input_file, self.gaz, self.word_alphabet,
                self.biword_alphabet, self.biword_count, self.char_alphabet,
                self.gaz_alphabet, self.gaz_count, self.gaz_split,
                self.label_alphabet, self.number_normalized,
                self.MAX_SENTENCE_LENGTH)
        else:
            print(
                "Error: you can only generate train/dev/test instance! Illegal input:%s"
                % (name))

    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print(
                "Error: illegal name during writing predict result, name should be within train/dev/test/raw !"
            )
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " +
                           predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" %
              (name, output_file))
