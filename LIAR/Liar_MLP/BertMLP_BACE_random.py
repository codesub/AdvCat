from __future__ import print_function
import logging
import argparse
import math

import torch.nn as nn

import time
from lm import NGramLangModel
from util import *
import spacy
import torch.nn.functional as F
import random
from tools import *
from copy import copy


from bert_serving.client import BertClient
client = BertClient()

nlp = spacy.load("en_core_web_lg")
NGRAM = 3
TAU = 0.5
N_NEIGHBOR = 100
N_REPLACE = 20 # len(doc)//N_REPLACE
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
    parser.add_argument('--alpha', default=8, type=int, help='exploitation number ')
    parser.add_argument('--model', default='LSTM', type=str, help='model: either CNN or LSTM')
    parser.add_argument('--train_path', action='store', default='./Dataset/BinaryClass/train.csv', type=str,
                        dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', default='./Dataset/BinaryClass/test_1.csv', type=str,
                        dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--output_path', default='./data/changed_lstm', type=str,
                        help='Path to output changed test.txt data')
    parser.add_argument('--embedding_path', default='./Dataset/paragram_300_sl999/paragram_300_sl999.txt', action='store',
                        dest='embedding_path',
                        help='Path to pre-trained embedding data')
    parser.add_argument('--model_path', action='store',
                        default='./Model_trained/BertMLP_lr_0.000010_b=4_LSTM_{276}.bak', dest='model_path',
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
        self.model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        if torch.cuda.is_available():
            self.model.to(device)
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.src_vocab, self.label_vocab = self.load_vocab(opt.train_path)
        # to compute the gradient, we need to set up the optimizer first
        self.criterion = nn.CrossEntropyLoss().to(device)

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
                sample_text = []
                for cand in candidates:
                    sample = [i for i in [x for x in cand if x.strip()]]
                    sample = client.encode([' '.join(sample)])
                    sample_text.append(sample[0])
                sample_text = torch.as_tensor(sample_text)
                preds = self.model(sample_text.cuda())
                preds = F.log_softmax(preds, dim=1)

                # print('pred_probs',pred_probs.shape,pred_probs)
                pred_probs = torch.exp(preds)

                # print('pred_probs', len(pred_probs))
                if pred_probs.shape[1] == 2:
                    pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                else:
                    result = (1-pred_probs[:,y]).max(dim = 0)
                    best_candidate_id = result[1].cpu().numpy().item()
                    pred_prob = result[0].cpu().detach().numpy().item()

                index_candidate[i] = best_candidate_id
                pred_candidate[i] = pred_prob

        return index_candidate,pred_candidate
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

    def Get_neigbor_list(self, words):
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

        return list_closest_neighbors


    def UCB(self, round, pred_set_list, N):
        mean = pred_set_list / N
        delta = np.sqrt(self.opt.alpha * math.log(round) / N)
        ucb = mean + delta

        return ucb

    def New_Word(self, words, arm_chain, list_closest_neighbors):
        new_word = copy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                if pos[1] != 0:
                    new_word[pos[0]] = list_closest_neighbors[pos[0]][pos[1] - 1]

        return new_word


    def attack(self, count, doc, y, NoAttack, log_f, N_REPLACE):
        # ---------------------------------word paraphrasing----------------------------------------------#
        words = doc.split()
        words_before = copy(words)

        sample = [i for i in [x for x in words if x.strip()]]
        sample = torch.from_numpy(client.encode([' '.join(sample)]))

        orig_prob, orig_pred = classify(sample.cuda(), self.model)
        orig_pred = orig_pred.data.cpu().numpy()
        pred, pred_prob = orig_pred, orig_prob
        if not (pred == y):  # attack success
            print(" this original samples predict wrong")
            NoAttack = NoAttack + 1
            print(pred, y)
            return [], 0, 0, 0,[], [orig_prob], 0,NoAttack,[],[pred],0
        best_score = 1 - pred_prob

        # now word level paraphrasing
        ## step1: find the neighbor for all the words.
        list_closest_neighbors = self.Get_neigbor_list(words)

        if self.opt.model == 'CNN':
            lword = min(len(words), self.model.sentence_len)
        else:
            lword = len(words)
        changed_words = []

        allCode_ori = list(range(len(words)))
        allCode = []
        for code_ind in allCode_ori:
            if len(list_closest_neighbors[code_ind]) != 0:
                allCode.append(code_ind)

        # if len(allCode) <= N_REPLACE:
        N_REP = len(allCode)
        # else:
        #     N_REP = N_REPLACE
        if N_REP < 1:
            print(" No words' neighbor can be changed")
            return [], 0, 0, 0,[], [orig_prob], 0,NoAttack,[],[orig_pred],0
        RandomSamp = N_REP
        RN = 5  # this is the time to repeat all the random process. to get a more stable result.
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        arm_preds = [pred_prob]
        arm_preds_class = [y]
        arm_chain_words = []

        for n in range(RN):
            # print("random index :",n)
            start_random = time.time()
            K_set = random.sample(allCode, RandomSamp)

            arm_chain = []  # the candidate set is S after selecting code process
            arm_pred = []
            arm_chain_word = []
            iteration = 0
            N = np.ones(len(K_set))
            time_Dur = 0
            robust_flag = 1
            new_words = copy(words)


            sample = [i for i in [x for x in new_words if x.strip()]]
            sample = torch.from_numpy(client.encode([' '.join(sample)]))

            orig_prob_new, orig_pred_new = classify(sample.cuda(), self.model)

            index_candidate, pred_set_list = self.word_paraphrase_nocombine(new_words, orig_pred_new, K_set,
                                                                             list_closest_neighbors, y)
            INDEX = []
            while robust_flag == 1 and len(arm_chain) <= Budget*len(allCode_ori) + 1 and time_Dur <= SECONDS:
                iteration += 1

                ucb = self.UCB(iteration, pred_set_list, N)
                topk_feature_index = np.argsort(ucb)[-1]
                INDEX.append(topk_feature_index)

                Feat_max = K_set[topk_feature_index]
                cand_max, pred_max = self.word_paraphrase_nocombine(new_words, orig_pred_new, [Feat_max],
                                                                                        list_closest_neighbors, y)

                new_words = self.New_Word(new_words, [(Feat_max, cand_max[0])], list_closest_neighbors)

                arm_chain.append((Feat_max, cand_max[0]))
                arm_pred.append(pred_max)

                n_add = np.eye(len(N))[topk_feature_index]
                N += n_add

                pred_set_list_add = np.zeros(len(K_set))
                pred_set_list_add[topk_feature_index] = pred_max
                pred_set_list = pred_set_list + pred_set_list_add

                time_end = time.time()
                time_Dur = time_end - start_random
                if arm_pred[-1] > TAU:
                    success.append(1)
                    num_armchain.append(len(arm_chain))
                    n_change.append(len(Counter(arm_chain).keys()))
                    time_success.append(time_Dur)
                    arm_chains.append(arm_chain)
                    arm_preds.append(arm_pred[-1].tolist()[0])
                    arm_preds_class.append(1-y)

                    robust_flag = 0

                    # print('arm_chain', arm_chain, file=log_f, flush=True)
                    # print('attack success', file=log_f, flush=True)
                    break



                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break
            if  robust_flag ==1:
                success.append(0)
                num_armchain.append(len(arm_chain))
                n_change.append(len(Counter(arm_chain).keys()))
                time_success.append(time_Dur)
                arm_chains.append(arm_chain)
                arm_preds.append(arm_pred[-1].tolist()[0])
                arm_preds_class.append(y)

            for i in range(len(words)):
                if words_before[i] != new_words[i]:
                    arm_chain_word.append([words_before[i], new_words[i]])
            arm_chain_words.append(arm_chain_word)
            if time_Dur > SECONDS:
                print('The time is over', time_Dur, file=log_f, flush=True)
                break

        return success, RN, num_armchain, time_success, arm_chains, arm_preds,np.array(n_change)/len(allCode_ori),NoAttack,arm_chain_words,arm_preds_class,RandomSamp


def main():
    opt = parse_args()
    log_f = open(
        './Logs/BertMLP/RSUCB_R_L_Budget_%s_N=%s_TAU=%s_N_REPLACE=%s_Time=%s_positive_updatesuccess_count.bak' % (
        str(Budget), str(N_NEIGHBOR), str(TAU), str(N_REPLACE), str(SECONDS)),
        'w+')

    TITLE = '=== ' + 'RSUCB_R' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + '_Time' + str(
        SECONDS) + ' ==='

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X_train, y_train = read_data_multi_class(opt.train_path)
    # X, y = read_data_multi_class(opt.test_path)
    [X, y] = pickle.load(open('./Dataset/BinaryClass/test_1_attack_all.pkl', 'rb'))
    attacker = Attacker(X_train, opt)
    del X_train
    del y_train

    success_num = 0
    success_num_count = 0
    NoAttack_num = 0
    F = []
    g = []
    F_V = []
    Total_iteration = 0
    Total_change = 0
    Total_time = 0
    Total_Query = 0
    i = 0
    changed_words_all = []

    testy_list = []
    yhat_classes_list = []
    yhat_probs_list = []

    for count, doc in enumerate(X):
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        success, RN, num_armchain, time_success, arm_chains, arm_preds,n_change,NoAttack_num,arm_chain_words,arm_preds_class,RandomSamp= attacker.attack(count, doc, y[count],
                                                                                         NoAttack_num, log_f, N_REPLACE)

        arm_chains_sample = []
        arm_preds_sample = 0
        success_pos = np.argwhere(np.array(success) ==1)
        fail_pos = np.argwhere(np.array(success) ==0)
        num_armchain_s = SuccessPosValue(num_armchain, success_pos)
        n_change_s= SuccessPosValue(n_change, success_pos)
        time_success_s = SuccessPosValue(time_success, success_pos)
        arm_preds_s = SuccessPosValue(arm_preds, success_pos+1)
        arm_preds_class_s= SuccessPosValue(arm_preds_class, success_pos+1)

        if len(success_pos) >= 1:
            # print('all random success attack in this sample')
            success_num_count +=1
            SR_sample = np.sum(success) / RN
            success_num += SR_sample
            AI_sample = np.average(num_armchain_s)
            Achange = np.average(n_change_s)
            AT_sample = np.average(time_success_s)

            changed_words_all.append(arm_chain_words)



            Total_iteration += AI_sample
            Total_change += Achange
            Total_time += AT_sample
            Total_Query = Total_Query + (RandomSamp * N_NEIGHBOR + AI_sample)


            testy_list += [y[count]] * len(success_pos)
            yhat_probs_list += np.array(arm_preds_s).tolist()
            yhat_classes_list += np.array(arm_preds_class_s).tolist()
            # print('success pos',testy_list,yhat_probs_list,yhat_classes_list)
            print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
            print(" Time: ", AT_sample, file=log_f, flush=True)

        elif len(success_pos) ==0:
            if len(fail_pos) ==0:
                # This is the attack fail condition
                if arm_preds[0] > TAU:
                    # This is no attack condition
                    testy_list += [y[count]]
                    yhat_probs_list += np.array(arm_preds).tolist()
                    yhat_classes_list += np.array(arm_preds_class).tolist()
                else:
                    # This is no attack condition under the original data is too short
                    testy_list += [y[count]]
                    yhat_probs_list += (np.array(arm_preds)).tolist()
                    yhat_classes_list += np.array(arm_preds_class).tolist()


        if len(fail_pos) > 0:
            arm_preds_f = SuccessPosValue(arm_preds, fail_pos + 1)
            arm_preds_class_f = SuccessPosValue(arm_preds_class, fail_pos + 1)
            testy_list += [y[count]]* len(fail_pos)
            yhat_probs_list += (1-np.array(arm_preds_f)).tolist()
            yhat_classes_list += (np.array(arm_preds_class_f)).tolist()
            # print('fail pos',testy_list,yhat_probs_list,yhat_classes_list)

        testy = np.array(testy_list)
        yhat_classes = np.array(yhat_classes_list)
        yhat_probs = np.array(yhat_probs_list)

        if len(np.unique(yhat_classes)) > 1 and len(np.unique(testy)) > 1:
            # print('performance',testy,yhat_classes,yhat_probs)
            performance(testy, yhat_classes, yhat_probs, log_f)


        print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
        print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (count - NoAttack_num) != 0 and success_num != 0:
            print("--- success Ratio: " + str(success_num / (count + 1 - NoAttack_num)) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Iteration: " + str(Total_iteration / success_num_count) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Query: " + str(Total_Query / success_num_count) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Change: " + str(Total_change / success_num_count) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Time: " + str(Total_time / success_num_count) + " ---", file=log_f,
                  flush=True)

            print("--- changed words: " + str(arm_chain_words) + " ---", file=log_f,
                  flush=True)

            pickle.dump(changed_words_all,
                        open("./Output/BertMLP/" + 'RSUCB_Budget=%s_N=%s_TAU=%s_N_REPLACE=%d_SECONDS=%s.pickle' % (
                        str(Budget), str(N_NEIGHBOR), str(TAU), N_REPLACE, str(SECONDS)), 'wb'))


if __name__ == '__main__':
    main()
