from __future__ import print_function
import logging
import argparse
import torch.nn as nn
import time
from lm import NGramLangModel
from util import *
import spacy
import torch.nn.functional as F
import random
from tools import *
from copy import copy

nlp = spacy.load("en_core_web_lg")
NGRAM = 3
TAU = 0.5
N_NEIGHBOR = 100
N_REPLACE = 0
SECONDS = 1000
Budget = 0.1
from gensim.models import KeyedVectors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_delta', default=2, type=int, help='percentage of allowed word paraphasing')
    parser.add_argument('--model', default='LSTM', type=str, help='model: either CNN or LSTM')
    parser.add_argument('--train_path', action='store', default='./Dataset/BinaryClass/train.csv', type=str,
                        dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', default='./Dataset/BinaryClass/test_1.csv', type=str,
                        dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--output_path', default='./Dataset/changed_lstm', type=str,
                        help='Path to output changed test.txt data')
    parser.add_argument('--embedding_path', default='./Dataset/paragram_300_sl999/paragram_300_sl999.txt', action='store',
                        dest='embedding_path',
                        help='Path to pre-trained embedding data')
    parser.add_argument('--model_path', action='store',
                        default='./Model_trained/W2VCNN_lr_0.000010_b=16_LSTM_{99}.bak', dest='model_path',
                        help='Path to pre-trained classifier model')
    parser.add_argument('--max_size', default=20000, type=int,
                        help='max amount of transformations to be processed by each iteration')
    parser.add_argument('--first_label', default='FAKE', help='The name of the first label that the model sees in the \
                         training data. The model will automatically set it to be the positive label. \
                         For instance, in the fake news dataset, the first label is FAKE.')

    return parser.parse_args()


class Attacker(object):
    ''' main part of the attack model '''

    def __init__(self, X, opt):
        self.opt = opt
        self.suffix = 'wordonly-' + str(opt.word_delta)
        self.DELTA_W = int(opt.word_delta) * 0.1
        self.TAU_2 = 2
        self.TAU_wmd_s = 0.75
        self.TAU_wmd_w = 0.75
        # want do sentence level paraphrase first
        X = [doc.split() for doc in X]
        logging.info("Initializing language model...")
        print("Initializing language model...")
        self.lm = NGramLangModel(X, NGRAM)
        logging.info("Initializing word vectors...")
        print("Initializing word vectors...")
        self.w2v = KeyedVectors.load_word2vec_format(opt.embedding_path, encoding='utf-8', unicode_errors='ignore')
        logging.info("Loading pre-trained classifier...")
        print("Loading pre-trained classifier...")
        # self.model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        self.model = torch.load(opt.model_path)
        if torch.cuda.is_available():
            self.model.to(device)
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.src_vocab, self.label_vocab = self.load_vocab(opt.train_path)
        # to compute the gradient, we need to set up the optimizer first
        self.criterion = nn.CrossEntropyLoss().to(device)

    def hidden(self, hidden_dim):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(1, 1, hidden_dim).to(device))
            c0 = Variable(torch.zeros(1, 1, hidden_dim).to(device))
        else:
            h0 = Variable(torch.zeros(1, 1, hidden_dim))
            c0 = Variable(torch.zeros(1, 1, hidden_dim))
        return (h0, c0)


    def forward_lstm(self, embed,
                     model):  # copying the structure of LSTMClassifer, just omitting the first embedding layer
        lstm_out, hidden0 = model.rnn(embed, self.hidden(512))
        y = model.linear(lstm_out[-1])
        return y

    def forward_cnn(self, embed, model):
        x_list = [conv_block(embed) for conv_block in model.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        return F.softmax(model.fc(out), dim=1)

    def text_to_var_CNN(self, docs, vocab):
        tensor = []
        max_len = self.model.sentence_len
        for doc in docs:
            vec = []
            for tok in doc:
                vec.append(vocab.stoi[tok])
            if len(doc) < max_len:
                vec += [0] * (max_len - len(doc))
            else:
                vec = vec[:max_len]
            tensor.append(vec)
        var = Variable(torch.LongTensor(tensor))
        if torch.cuda.is_available():
            var = var.to(device)
        return var

    def text_to_var(self,docs, vocab):
        tensor = []
        max_len = max([len(doc) for doc in docs])
        for doc in docs:
            vec = []
            for tok in doc:
                vec.append(vocab.stoi[tok])
            if len(doc) < max_len:
                vec += [vocab.stoi['<pad>']] * (max_len - len(doc))
            tensor.append(vec)
        try:
            var = Variable(torch.LongTensor(tensor)).transpose(0, 1)
        except:
            print(tensor, docs)
            exit()
        if torch.cuda.is_available():
            var = var.cuda()
        return var

    def load_vocab(self, path):
        src_field = data.Field()
        label_field = data.Field(pad_token=None, unk_token=None)
        dataset = data.TabularDataset(
            path=path, format='csv',
            fields=[('text', src_field), ('label', label_field)]
        )
        src_field.build_vocab(dataset, max_size=100000, min_freq=2, vectors="glove.6B.300d")
        label_field.build_vocab(dataset)
        return src_field.vocab, label_field.vocab


    def New_Word(self,words,arm_chain,list_closest_neighbors):
        new_word = copy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                if pos[1] != 0:
                    new_word[pos[0]] = list_closest_neighbors[pos[0]][pos[1] - 1]
        return new_word

    def word_paraphrase_nocombine(self, new_words,orig_pred_new, poses, list_closest_neighbors, y):
        index_candidate = np.zeros(len(poses), dtype=int)
        pred_candidate = np.zeros(len(poses))
        for i,candi_sample_index in enumerate(poses):
            candidates = [new_words]
            for neibor_index, neibor in enumerate(
                    list_closest_neighbors[candi_sample_index]):  # it omit the [] empty list

                corrupted = copy(new_words)
                corrupted[candi_sample_index] = neibor
                candidates.append(corrupted)
            if len(candidates) != 1:
                candidate_var = text_to_var_CNN(candidates, self.src_vocab)
                pred_probs = self.model(candidate_var)
                pred_probs = F.log_softmax(pred_probs, dim=1)

                # print('pred_probs',pred_probs.shape,pred_probs)
                pred_probs = torch.exp(pred_probs)
                # print('pred_probs', len(pred_probs))
                if pred_probs.shape[1] == 2:
                    pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                else:
                    result = (1 - pred_probs[:, y]).max(dim=0)
                    best_candidate_id = result[1].cpu().numpy().item()
                    pred_prob = result[0].cpu().detach().numpy().item()

                index_candidate[i] = best_candidate_id
                pred_candidate[i] = pred_prob

        return index_candidate, pred_candidate


    def WordNeigbor_select(self,new_words, orig_pred_new, K_set,list_closest_neighbors,y,arm_chain):
        index_candidate, pred_candidate= self.word_paraphrase_nocombine(new_words, orig_pred_new, K_set,
                                                                                     list_closest_neighbors,y)

        topk_feature_index = np.argsort(pred_candidate)[-1]

        pred_max = pred_candidate[topk_feature_index]
        Feat_max = K_set[topk_feature_index]
        cand_max = index_candidate[topk_feature_index]



        return (Feat_max, cand_max),pred_max


    def attack(self, count, doc, y, NoAttack,log_f):
        # ---------------------------------word paraphrasing----------------------------------------------#
        words = doc.split()
        words_before = copy(words)

        doc_var = text_to_var_CNN([words], self.src_vocab)

        orig_prob, orig_pred = classify(doc_var, self.model)
        orig_pred = orig_pred.data.cpu().numpy()
        pred, pred_prob = orig_pred, orig_prob
        if not (pred == y):  # attack success
            print(" this original samples predict wrong")
            NoAttack = NoAttack + 1
            print(pred, y)
            return words,1-pred_prob , 0, 0, 0, 0, NoAttack, 0
        best_score = 1 - pred_prob

        # now word level paraphrasing
        ## step1: find the neighbor for all the words.
        list_closest_neighbors = []
        for pos, w in enumerate(words):
            if self.opt.model == 'CNN' and pos >= self.model.sentence_len: break
            try:
                closest_neighbors = self.w2v.most_similar(positive=[w.lower()],
                                                          topn=N_NEIGHBOR)  # get the 15 neighbors as the replacement set for this word.
            except:
                closest_neighbors = []
            closest_paraphrases = []
            closest_paraphrases.extend(closest_neighbors)
            # check if the words make sense
            valid_paraphrases = []
            doc1 = nlp(w)
            for repl, repl_sim in closest_paraphrases:
                doc2 = nlp(repl)  # ' '.join(repl_words))
                score = doc1.similarity(doc2)
                syntactic_diff = self.lm.log_prob_diff(words, pos, repl)
                logging.debug("Syntactic difference: %f", syntactic_diff)
                if score >= self.TAU_wmd_w and syntactic_diff <= self.TAU_2:  # check the chosen word useful or not
                    valid_paraphrases.append(repl)
            list_closest_neighbors.append(valid_paraphrases)  # closest_neighbors)
            if not closest_paraphrases:  # neighbors:
                # print('find no neighbor for word: '+w)
                continue


        changed_pos = set()
        iteration = 0
        recompute = True
        n_change = 0
        if self.opt.model == 'CNN':
            lword = min(len(words), self.model.sentence_len)
        else:
            lword = len(words)

        allCode_ori = list(range(len(words)))
        allCode = []
        for code_ind in allCode_ori:
            if len(list_closest_neighbors[code_ind]) != 0:
                allCode.append(code_ind)

        allCode = list(range(len(words)))
        Set_candi_sample = []
        iteration = 0
        query_num = 0
        words_before = copy(words)
        changed_words = []

        Set_c = [()]
        Set_best = []
        Set_residual = DiffElem(allCode, Set_best)
        pred_best = best_score
        robust_flag =1
        ## step2: this is attack main part.
        st = time.time()
        while robust_flag == 1 and n_change <= Budget * len(allCode_ori) + 1 and time.time() - st <= SECONDS:  # no attack success condition only 0.2 original feature
            iteration += 1

            Pred_select = []
            Set_select = []
            arm_chain = []
            for set_ in Set_c:  # all the combination set
                if len(set_)==0:
                    new_words = copy(words)
                else:
                    new_words = copy(words)
                    new_words = self.New_Word(new_words,set_,list_closest_neighbors)

                new_words_var = text_to_var_CNN([new_words], self.src_vocab)
                orig_prob_new, orig_pred_new = classify(new_words_var, self.model)

                (Feat_max, cand_max), pred_max = self.WordNeigbor_select(new_words, orig_pred_new,Set_residual,
                                                                            list_closest_neighbors, y, set_ )


                if pred_max >= best_score:
                    Set_select.append(set_)
                    arm_chain.append((Feat_max, cand_max))
                    Pred_select.append(pred_max)
                    best_score = pred_max

            if len(arm_chain) != 0:
                pred_best = Pred_select[-1]
                new_words = self.New_Word(new_words, [arm_chain[-1]], list_closest_neighbors)
                Set_best.append(arm_chain[-1])
            else:
                pred_best = best_score
                new_words = new_words
                Set_best.append((Feat_max, cand_max))

            n_change = sum([0 if words_before[i] == new_words[i] else 1 for i in range(len(words))])
            Set_residual = DiffElem(allCode, [Set_best[-1][0]])
            query_num += len(Set_residual) *(len(Set_c))*N_NEIGHBOR
            Set_c = list(powerset(Set_best))

            if pred_best >= TAU:
                robust_flag = 0

        Time = time.time() - st
        for i in range(len(words)):
            if words_before[i] != new_words[i]:
                changed_words.append([words_before[i],new_words[i]])

        return new_words, pred_best, n_change/len(allCode_ori), changed_words, Time,  iteration, NoAttack,query_num

def main():
    opt = parse_args()
    log_f = open(
        './Logs/W2VCNN/FSGS_Budget=%s_N=%s_TAU=%s_N_REPLACE=%d_SECONDS=%s.bak' % (str(Budget),str(N_NEIGHBOR), str(TAU), N_REPLACE,str(SECONDS)),
        'w+')

    TITLE = '=== ' + 'FSGS' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + ' ==='

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X_train, y_train = read_data_multi_class(opt.train_path)
    # X, y = read_data_multi_class(opt.test_path)
    [X, y] = pickle.load(open('./Dataset/BinaryClass/test_1_attack_all.pkl', 'rb'))
    attacker = Attacker(X_train, opt)
    del X_train
    del y_train
    suc = 0
    time = 0
    Num_change = 0
    Iter = 0
    Query = 0
    NoAttack = 0
    noattack = 0
    changed_words_all = []

    yhat_classes_list = []
    yhat_probs_list = []
    testy_list = []

    for count, doc in enumerate(X):
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        changed_doc, flag, num_changed, changed_words, Time, iteration, NoAttack,query_num = attacker.attack(count, doc,
                                                                                                         y[count],
                                                                                                         NoAttack,log_f)
        print(" The cost Time", Time, file=log_f, flush=True)
        print(" The number changed", num_changed, file=log_f, flush=True)
        print(" The changed words", changed_words, file=log_f, flush=True)
        print(" NoAttack Number", NoAttack, file=log_f, flush=True)

        v = float(flag)
        if v > TAU:
            classes = 1 - y[count]
            suc += 1
            time = time + Time
            Num_change = Num_change + num_changed
            Iter = Iter + iteration
            Query = Query + query_num

            yhat_probs_list.append(v)
            yhat_classes_list.append(classes)
            # print('success',y[count],classes,yhat_probs_list,yhat_classes_list)
        elif v <= TAU:
            if NoAttack > noattack :
                # this is the no attack
                noattack = NoAttack
                classes = 1-y[count]
                yhat_probs_list.append(1-v)
                yhat_classes_list.append(classes)
                # print('fail', y[count], classes, yhat_probs_list, yhat_classes_list)
            else:
                # This is attack fail
                classes = y[count]
                yhat_probs_list.append(1-v)
                yhat_classes_list.append(classes)
                # print('fail', y[count], classes, yhat_probs_list, yhat_classes_list)


        yhat_classes = np.array(yhat_classes_list)
        yhat_probs = np.array(yhat_probs_list)

        testy_list.append(y[count])
        testy = np.array(testy_list)
        if len(np.unique(yhat_classes))!=1:
            performance(testy,yhat_classes,yhat_probs,log_f)

        if suc != 0 and ((count + 1) - NoAttack) != 0:
            SR = suc / ((count + 1) - NoAttack)
            AverTime = time / suc
            AverChanged = Num_change / suc
            AverIter = Iter/suc
            AverQuery = Query/suc
            print("SuccessRate = %f, AverTime = %f, AverChangedRatio = %f,AverIter = %f,AverQuery = %fchanged_words: %s" % (
                SR, AverTime, AverChanged,AverIter,AverQuery, str(changed_words)), file=log_f, flush=True)
            changed_words_all.append(changed_words)
            pickle.dump(changed_words_all,
                        open( "./Output/W2VCNN/"+ 'FSGS_Budget=%s_N=%s_TAU=%s_N_REPLACE=%d_SECONDS=%s.pickle' % (str(Budget),str(N_NEIGHBOR), str(TAU), N_REPLACE,str(SECONDS)), 'wb'))


if __name__ == '__main__':
    main()
