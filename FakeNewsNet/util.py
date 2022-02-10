import numpy as np
import pickle
import os
import copy
from math import log
from collections import Counter
import torch.utils.data
from nltk.util import ngrams
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix



class NGramLangModel(object):
    def __init__(self, corpus, n):
        self.default = 0.0001
        self.n = n
        self.ngram_counter = [None] * n
        for i in range(n, 0, -1):
            self.ngram_counter[i - 1] = NGramLangModel.get_ngram(corpus, i)

    @staticmethod
    def get_ngram(corpus, n):
        counter = Counter()
        for doc in corpus:
            counter.update(ngrams(doc, n))
        return counter

    def log_prob(self, x, n):
        if x[0] is None:
            return self.log_prob(x[1:], n - 1)
        elif x[-1] is None:
            return self.log_prob(x[:-1], n - 1)
        else:
            logp = log(self.ngram_counter[n - 1].get(tuple(x), self.default))
            # print(logp)
            logp -= log(self.ngram_counter[n - 2].get(tuple(x[:-1]), self.default))
            # print(',',logp)
            return logp

    def log_prob_diff(self, sentence, pos, repl):
        pad = [None] * (self.n - 1)
        padded_sent = pad + sentence + pad

        window_words = padded_sent[pos:pos + 2 * self.n - 1]
        repl_window_words = copy.copy(window_words)
        repl_window_words[self.n - 1] = repl

        log_p = 0
        repl_log_p = 0
        for i in range(self.n, 2 * self.n):
            repl_log_p += self.log_prob(repl_window_words[i - self.n:i], self.n)
            log_p += self.log_prob(window_words[i - self.n:i], self.n)

        return (log_p - repl_log_p) / self.n

    def log_prob_alone(self, sentence):
        pad = [None] * (self.n - 1)
        padded_sent = pad + sentence + pad
        log_p = 0
        for pos in range(len(sentence)):
            window_words = padded_sent[pos:pos + 2 * self.n - 1]
            for i in range(self.n, 2 * self.n):
                log_p += self.log_prob(window_words[i - self.n:i], self.n)
        if len(sentence):
            return log_p / len(sentence)
        else:
            return 0


# make sure the path exist
def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)


def write_file(Dataset, Model_Type, model_name, budget, algorithm, time_limit):
    log_f = open('./Logs/%s/%s/%s/MF_%s_%s.bak' % (Dataset, Model_Type, model_name, algorithm, str(budget)), 'w+')
    TITLE = '=== ' + Dataset + Model_Type + str(budget) + algorithm + ' time = ' + str(time_limit) + ' ==='
    print(TITLE, file=log_f, flush=True)
    directory = './Logs/%s/%s/%s/%s/' % (Dataset, Model_Type, model_name, algorithm)
    print()
    print(directory)
    print(directory, file=log_f, flush=True)
    Algorithm = directory
    mf_process_temp = pickle.load(open(Algorithm + 'mf_process_%s.pickle' % str(budget), 'rb'))
    changed_set_process_temp = pickle.load(open(directory + 'changed_set_process_%s.pickle' % str(budget), 'rb'))
    robust_flag = pickle.load(open(directory + 'robust_flag_%s.pickle' % str(budget), 'rb'))
    query_num = pickle.load(open(directory + 'querynum_%s.pickle' % str(budget), 'rb'))
    time = pickle.load(open(directory + 'time_%s.pickle' % str(budget), 'rb'))
    flip_percent = pickle.load(open(directory + 'flip_percent_%s.pickle' % str(budget), 'rb'))
    iteration_file = pickle.load(open(directory + 'iteration_%s.pickle' % str(budget), 'rb'))
    mf_process = []
    changed_set_process = []
    time_attack = []
    query_num_attack = []
    flip_changed_num = []
    iteration = []
    attack_num = 0
    for j in range(len(robust_flag)):
        if robust_flag[j] == 0:
            mf_process.append(mf_process_temp[j])
            changed_set_process.append(changed_set_process_temp[j])
            time_attack.append(time[j])
            query_num_attack.append(query_num[j])
            flip_changed_num.append(len(changed_set_process_temp[j][-1]))
            iteration.append(iteration_file[j])
        if robust_flag[j] != -1:
            attack_num += 1

    sorted_flip_changed_num = np.sort(flip_changed_num)
    change_medium = sorted_flip_changed_num[len(flip_changed_num) // 2]

    print('success rate:', len(iteration) / attack_num)
    print('average iteration:', np.mean(iteration))
    print('average changed code', np.mean(flip_changed_num))
    print('average time:', np.mean(time_attack))
    print('average query number', np.mean(query_num_attack))
    print('medium changed number', change_medium)
    print('flip_ratio:', np.around(np.mean(flip_percent), decimals=3))

    print('success rate:', len(iteration) / attack_num, file=log_f, flush=True)
    print('average iteration:', np.mean(iteration), file=log_f, flush=True)
    print('average changed code', np.mean(flip_changed_num), file=log_f, flush=True)
    print('average time:', np.mean(time_attack), file=log_f, flush=True)
    print('average query number', np.mean(query_num_attack), file=log_f, flush=True)
    print('medium changed number', change_medium, file=log_f, flush=True)
    print('flip_ratio:', np.around(np.mean(flip_percent), decimals=3), file=log_f, flush=True)

    print('end')
    print()
    print()
    print()


class Pol_Dataset(Dataset):
    def __init__(self, x, y, args, mode, need_neighbor=False):
        self.data = x
        try:
            self.x = torch.FloatTensor(x)
        except ValueError:
            max_len = 50
            self.x = torch.tensor([])
            for i in range(len(x)):
                text_vector = torch.FloatTensor(x[i])
                if len(text_vector) > max_len:
                    text_vector = text_vector[:max_len].unsqueeze(0)
                else:
                    zeros = torch.zeros(max_len - len(x[i]), text_vector.size(1))
                    text_vector = torch.cat((text_vector, zeros), dim=0).unsqueeze(0)
                self.x = torch.cat((self.x, text_vector), dim=0)
        self.y = torch.LongTensor(y)
        if args.feature == 'spacy':
            self.num_features = 300
        else:
            self.num_features = 768
        self.num_classes = 2
        self.need_neighbor = need_neighbor
        test_idx = pickle.load(open('./politifact/test_idx.pickle', 'rb'))
        train_idx = pickle.load(open('./politifact/train_idx.pickle', 'rb'))
        val_idx = pickle.load(open('./politifact/val_idx.pickle', 'rb'))

        if mode == 'train':
            self.x = self.x[train_idx]
            self.y = self.y[train_idx]
        elif mode == 'val':
            self.x = self.x[val_idx]
            self.y = self.y[val_idx]
        elif mode == 'test':
            self.x = self.x[test_idx]
            self.y = self.y[test_idx]
        elif args.data_type == 'perturbed':
            self.y = self.y[test_idx]
        else:
            pass

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.need_neighbor:
            return x, y, idx
        return x, y, len(x)


def eval_deep(log, loader):
    """
    Evaluating the classification performance given mini-batch data
    """

    # get the empirical batch_size for each mini-batch
    data_size = loader.dataset.__len__()
    batch_size = loader.batch_size
    if data_size % batch_size == 0:
        size_list = [batch_size] * (data_size // batch_size)
    else:
        size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size]

    assert len(log) == len(size_list)

    accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

    prob_log, label_log = [], []
    CM = 0
    for batch, size in zip(log, size_list):
        pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
        prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
        label_log.extend(y)

        accuracy += accuracy_score(y, pred_y) * size
        f1_macro += f1_score(y, pred_y, average='macro') * size
        f1_micro += f1_score(y, pred_y, average='micro') * size
        precision += precision_score(y, pred_y, zero_division=0) * size
        recall += recall_score(y, pred_y, zero_division=0) * size

        CM += confusion_matrix(y, pred_y)
        print(CM)

    auc = roc_auc_score(label_log, prob_log)
    FNR = CM[1][0] / (CM[1][0] + CM[1][1])
    FPR = CM[0][1] / (CM[0][1] + CM[0][0])

    return accuracy / data_size, f1_macro / data_size, f1_micro / data_size, precision / data_size, recall / data_size, FNR, FPR, auc