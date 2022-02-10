import logging
import argparse
import math
from copy import copy
import time
import random
from collections import Counter
import os
import copy
from utils import *
from models import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrep', default=5, type=int,
                        help='N_Replace')
    parser.add_argument('--alpha', default=4, type=int,
                        help='alpha')
    parser.add_argument('--budget', default=5, type=int, help='purturb budget')
    parser.add_argument('--dataset', default='tiresias', type=str, help='dataset')
    parser.add_argument('--Dtype', default='normal', type=str, help='test_dataset_type')
    parser.add_argument('--P_k', default='9', type=int, help='pred_top-k')
    parser.add_argument('--D_k', default='1', type=int, help='grouped by k times for training')
    parser.add_argument('--a', default='0', type=int, help='start fo training')
    parser.add_argument('--b', default='1', type=int, help='end of training')
    parser.add_argument('--modeltype', default='Normal', type=str, help='model type')
    parser.add_argument('--time', default=60, type=int, help='time limit')
    parser.add_argument('--rand_k', default=100, type=int,
                        help='randomly choose k categories in word_paraphrase_nocombine')
    parser.add_argument('--k_loop', default=5, type=int,
                        help='loop k times for each sample to find average ucb attack performance')

    return parser.parse_args()


class Attacker(object):
    ''' main part of the attack model '''

    def __init__(self, best_parameters_file, opt):
        self.opt = opt
        if opt.dataset == 'IPS':
            self.model = IPSRNN()
        elif opt.dataset == 'hdfs':
            self.model = DeepCase()
        elif opt.dataset == 'tiresias':
            self.model = Tiresias_DeepCase()
        else:
            self.model = None
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(best_parameters_file))
        self.model.eval()
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.criterion = nn.CrossEntropyLoss().to(device)

    def word_paraphrase_nocombine(self, new_words, orig_pred_new, poses, list_closest_neighbors, y):
        index_candidate = np.zeros(len(poses), dtype=int)
        pred_candidate = np.zeros(len(poses))

        for i, candi_sample_index in enumerate(poses):
            candidates = []
            best_pred_prob = -1.0
            neighbors = np.random.choice(list_closest_neighbors, self.opt.rand_k)
            for neibor in neighbors:
                corrupted = copy.deepcopy(new_words)
                corrupted[candi_sample_index] = neibor
                candidates.append(corrupted)
            if len(candidates) != 1:
                batch_size = 32
                n_batches = int(np.ceil(float(len(candidates)) / float(batch_size)))
                for index in range(n_batches):
                    batch_diagnosis_codes = candidates[batch_size * index: batch_size * (index + 1)]
                    batch_labels = [y] * len(batch_diagnosis_codes)
                    t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[self.opt.dataset])
                    t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
                    pred_probs = self.model(t_diagnosis_codes)
                    subsets_h, _ = pred_probs.topk(P_k + 1)
                    subsets_h = subsets_h[:, -1]
                    subsets_h = subsets_h.detach().cpu().numpy()
                    pred_probs = pred_probs.data.cpu().numpy()
                    subsets_g = pred_probs[:, y]
                    subsets_object = subsets_h - subsets_g
                    result = subsets_object.max(axis=0)
                    result_index = subsets_object.argmax(axis=0)
                    candidate_id = result_index.item() + batch_size * index
                    pred_prob = result.item()

                    if pred_prob > best_pred_prob:
                        best_pred_prob = pred_prob
                        index_candidate[i] = neighbors[candidate_id]
                        pred_candidate[i] = pred_prob

        return index_candidate, pred_candidate

    def UCB(self, round, pred_set_list, N):

        mean = pred_set_list / N
        delta = np.sqrt(self.opt.alpha * math.log(round) / N)
        ucb = mean + delta

        return ucb

    def New_Word(self, words, arm_chain, list_closest_neighbors):
        new_word = copy.deepcopy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                new_word[pos[0]] = list_closest_neighbors[pos[1]]
        return new_word

    def input_handle(self, funccall, y):
        funccall = [funccall]
        y = [y]
        t_diagnosis_codes, _ = pad_matrix(funccall, y, num_category[self.opt.dataset])
        return torch.tensor(t_diagnosis_codes).cuda()

    def classify(self, funccall, y):
        weight_of_embed_codes = self.input_handle(funccall, y)
        logit = self.model(weight_of_embed_codes)
        g_k, pred_k = logit.topk(P_k)
        h, _ = logit.topk(P_k + 1)
        h = h[:, -1]
        h = h.detach().cpu().numpy()
        logit = logit.data.cpu().numpy()
        g = logit[:, y]
        pred_k = pred_k.squeeze().detach().cpu().numpy().tolist()
        return pred_k, g, h, logit

    def attack(self, count, words, y, NoAttack, log_f, N_REPLACE):
        SECONDS = self.opt.time
        Budget = self.opt.budget
        print('The original words is:', words, file=log_f, flush=True)
        orig_pred, orig_g, orig_h, orig_index = self.classify(words, y)
        print(" the original index is:", orig_index, file=log_f, flush=True)
        pred, pred_g = orig_pred, orig_g
        if not y in pred:
            print(" this original samples predict wrong", file=log_f, flush=True)
            NoAttack = NoAttack + 1
            print(pred, y, file=log_f, flush=True)
            return 0, 0, [], 0, 0, 0, 0, 0, NoAttack
        list_closest_neighbors = range(num_category[self.opt.dataset])
        lword = num_feature[self.opt.dataset]
        allCode = range(lword)
        if len(allCode) <= N_REPLACE:
            N_REP = len(allCode)
        else:
            N_REP = N_REPLACE

        RN = self.opt.k_loop
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        query_nums = []
        arm_preds = [pred_g]

        for n in range(RN):
            print("random index :",n,file=log_f, flush=True)
            start_random = time.time()
            K_set = random.sample(allCode, N_REP)
            arm_chain = []
            arm_pred = []
            iteration = 0
            N = np.ones(len(K_set))
            time_Dur = 0
            robust_flag = 1
            new_words = copy.deepcopy(words)
            orig_pred_new, orig_g_new, orig_h_new, index_new= self.classify(new_words, y)
            index_candidate, pred_set_list = self.word_paraphrase_nocombine(new_words, orig_pred_new, K_set,
                                                                            list_closest_neighbors, y)
            print('index_candidate', index_candidate, pred_set_list, file=log_f, flush=True)
            INDEX = []
            while robust_flag == 1 and len(Counter(arm_chain).keys()) <= Budget and time_Dur <= SECONDS:
                iteration += 1

                ucb = self.UCB(iteration, pred_set_list, N)
                topk_feature_index = np.argsort(ucb)[-1]
                INDEX.append(topk_feature_index)

                Feat_max = K_set[topk_feature_index]
                cand_max, pred_max = self.word_paraphrase_nocombine(new_words, orig_pred_new, [Feat_max],
                                                                    list_closest_neighbors, y)
                print('cand_max, pred_max', (cand_max, pred_max), file=log_f, flush=True)

                new_words = self.New_Word(new_words, [(Feat_max, cand_max[0])], list_closest_neighbors)
                print('The new_words is:', new_words, file=log_f, flush=True)

                arm_chain.append((Feat_max, cand_max[0]))
                arm_pred.append(pred_max)

                n_add = np.eye(len(N))[topk_feature_index]
                N += n_add

                pred_set_list_add = np.zeros(len(K_set))
                pred_set_list_add[topk_feature_index] = pred_max
                pred_set_list = pred_set_list + pred_set_list_add

                time_end = time.time()
                time_Dur = time_end - start_random
                print('arm_chain', arm_chain, file=log_f, flush=True)
                print('arm_pred', arm_pred, file=log_f, flush=True)
                if arm_pred[-1] > 0:
                    success.append(1)
                    num_armchain.append(len(arm_chain))
                    n_change.append(len(Counter(arm_chain).keys()))
                    time_success.append(time_Dur)
                    arm_chains.append(arm_chain)
                    arm_preds.append(arm_pred)
                    query_nums.append(iteration * (N_REP * self.opt.rand_k + 1))
                    robust_flag = 0
                    break
                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break
            if time_Dur > SECONDS:
                print('The time is over', time_Dur, file=log_f, flush=True)
                break

        return success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change, query_nums, NoAttack


def main():
    output_file = './Logs/%s/%s/%s/%s/%s/' % (att_name, Dataset, Dtype,P_k,D_k)
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    log_f = open(
        './Logs/%s/%s/%s/%s/%s/Budget_%s_ALPHA=%s_N_REPLACE=%s_Time=%s_Loop_%s.bak' % (
            att_name, Dataset, Dtype, P_k,D_k,str(Budget), str(alpha), str(N_REPLACE), str(SECONDS), str(loop)), 'w+')

    TITLE = '=== ' + 'UCB_R:' + ' target_mf = ' + str(0) + ' changeRange = ' + str(N_REPLACE) + ' Time' + str(
        SECONDS) + ' Loop=' + str(loop) + ' ==='

    best_parameters_file = 'Deepcase_tiresias.pt'

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X, y = load_data(Dataset, Dtype, a, b)
    attacker = Attacker(best_parameters_file, opt)

    success_num = 0
    NoAttack_num = 0
    Total_iteration = 0
    Total_change = 0
    Total_time = 0
    Total_query_num = 0

    for count, doc in enumerate(X):
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change, query_nums, NoAttack_num = \
            attacker.attack(count, doc, y[count], NoAttack_num, log_f, N_REPLACE)

        if np.sum(success) >= 1:
            SR_sample = np.sum(success) / RN
            success_num += SR_sample
            AI_sample = np.average(num_armchain)
            Achange = np.average(n_change)
            AT_sample = np.average(time_success)
            query_num = np.average(query_nums)

            Total_iteration += AI_sample
            Total_change += Achange
            Total_time += AT_sample
            Total_query_num += query_num

            print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
            print(" Time: ", AT_sample, file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
        print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (count - NoAttack_num) != 0 and success_num != 0:
            print("--- success Ratio: " + str(success_num / (count + 1 - NoAttack_num)) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Change: " + str(Total_change / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Query Num: " + str(Total_query_num / success_num) + " ---", file=log_f,
                  flush=True)

num_classes = {
    'Splice': 3,
    'IPS': 3,
    'hdfs': 28,
     'tiresias': 1113,}
opt = parse_args()
Dataset = opt.dataset
Dtype = opt.Dtype
P_k = opt.P_k
D_k = opt.D_k
a = opt.a
b = opt.b 
SECONDS = opt.time
Budget = opt.budget
Data_Type = opt.modeltype
N_REPLACE = opt.nrep
alpha = opt.alpha
loop = opt.k_loop
att_name = 'UCB'
if __name__ == '__main__':
    main()
