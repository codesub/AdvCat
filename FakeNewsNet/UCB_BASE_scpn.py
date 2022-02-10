import math
import time
import argparse
import spacy
import random
from models import *
from util import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrep', default=40, type=int,
                        help='N_Replace')
    parser.add_argument('--alpha', default=4, type=int,
                        help='alpha')
    parser.add_argument('--budget', default=0.2, type=float, help='purturb budget')
    parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--time', default=600, type=int, help='time limit')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--feature', type=str, default='bert', help='feature type, [spacy, bert]')
    parser.add_argument('--model', type=str, default='mlp', help='model type, [mlp, lstm]')
    parser.add_argument('--data_type', type=str, default='clean', help='data type, [clean, perturbed]')
    parser.add_argument('--k_loop', default=1, type=int,
                        help='loop k times for each sample to find average ucb attack performance')

    return parser.parse_args()


class Attacker(object):
    def __init__(self, args, best_parameters_file):
        self.args = args
        if args.model == 'mlp':
            self.model = MLP(args)
        else:
            self.model = LSTM(args)
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.model = self.model.to(args.device)
        self.model.eval()
        print("Initializing vocabularies...")
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.nlp = spacy.load("en_core_web_lg")

    def word_paraphrase_nocombine(self, new_words, poses, length, list_closest_neighbors, y, new_data):
        index_candidate = []
        pred_candidate = np.zeros(len(poses)) - 1
        data_collect = []

        for i, candi_sample_index in enumerate(poses):
            index_candidate.append(new_words[candi_sample_index])
            candidates = []
            data_cand = torch.tensor([])
            best_pred_prob = -1.0
            data_collect.append(copy.deepcopy(new_data.tolist()))
            for neighbor in list_closest_neighbors[candi_sample_index]:
                corrupted = copy.deepcopy(new_words)
                temp_data = copy.deepcopy(new_data)

                corrupted[candi_sample_index] = neighbor

                if self.args.model == 'mlp':
                    text = ' '.join(corrupted)
                    if self.args.feature == 'spacy':
                        doc_vec = self.nlp(text).vector
                        temp_data = torch.from_numpy(doc_vec)
                    else:
                        encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
                        embedding = bert(**encoded_input)
                        temp_data = embedding[1][0].data
                else:
                    if self.args.feature == 'spacy':
                        doc_vec = self.nlp(neighbor).vector
                        temp_data[candi_sample_index] = torch.from_numpy(doc_vec)

                    else:
                        encoded_input = tokenizer(neighbor, return_tensors='pt', max_length=512, truncation=True)
                        embedding = bert(**encoded_input)
                        temp_data[candi_sample_index] = embedding[1][0].data
                temp_data = temp_data.unsqueeze(0)

                data_cand = torch.cat((data_cand, temp_data), dim=0)
                candidates.append(corrupted)
            data_cand = torch.cat((data_cand, new_data.unsqueeze(0)), dim=0)
            candidates.append(new_words)
            if len(candidates) != 1:
                batch_size = self.args.batch_size
                n_batches = int(np.ceil(float(len(candidates)) / float(batch_size)))
                for index in range(n_batches):
                    data_temp = data_cand[batch_size * index: batch_size * (index + 1)].to(self.args.device)
                    if self.args.model == 'mlp':
                        logit = self.model(data_temp)
                    else:
                        logit = self.model(data_temp, [length] * len(data_temp))
                    logit = F.softmax(logit, dim=1)
                    pred_probs = logit.cpu().detach().numpy()
                    subsets_g = pred_probs[:, y]
                    subsets_h = 1 - subsets_g
                    result = (subsets_h - subsets_g).max(axis=0)
                    result_index = (subsets_h - subsets_g).argmax(axis=0)
                    candidate_id = result_index.item() + batch_size * index
                    pred_prob = result.item()

                    if pred_prob > best_pred_prob:
                        best_pred_prob = pred_prob
                        if candidate_id == len(candidates) - 1:
                            index_candidate[i] = new_words[candi_sample_index]
                        else:
                            index_candidate[i] = list_closest_neighbors[candi_sample_index][candidate_id]
                            data_collect[i] = data_cand[candidate_id].tolist()
                        pred_candidate[i] = pred_prob
        return index_candidate, pred_candidate, data_collect

    def UCB(self, round, pred_set_list, N):

        mean = pred_set_list / N
        delta = np.sqrt(self.args.alpha * math.log(round) / N)
        ucb = mean + delta

        return ucb

    def New_Word(self, words, arm_chain):
        new_word = copy.deepcopy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                new_word[pos[0]] = pos[1]
        return new_word

    def classify(self, x, y, length):
        data = copy.deepcopy(x)
        data = data.unsqueeze(0)
        data = data.to(self.args.device)
        y = y.to(self.args.device)
        if self.args.model == 'mlp':
            logit = self.model(data)
        else:
            logit = self.model(data, [length])
        logit = F.softmax(logit, dim=1)
        logit = logit.cpu()
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
        logit = logit.data.cpu().numpy()
        g = logit[0][y]
        h = 1 - g
        return pred[0], g, h

    def attack(self, count, data, text, y, NoAttack, log_f, N_REPLACE, sentence_neighbors, sentences, length):
        SECONDS = self.args.time
        Budget = self.args.budget
        orig_pred, orig_g, orig_h = self.classify(data, y, length)
        pred, pred_g = orig_pred, orig_g
        if not (pred == y):
            print(" this original samples predict wrong", file=log_f, flush=True)
            NoAttack = NoAttack + 1
            print(pred, y, file=log_f, flush=True)
            return 0, 0, [], 0, 0, 0, 0, 0, NoAttack, 1, data.tolist()

        num_sent = len(sentences)
        list_closest_neighbors = sentence_neighbors

        lword = len(sentences)

        allCode = range(min(lword, 50))

        if len(allCode) <= N_REPLACE:
            N_REP = len(allCode)
        else:
            N_REP = N_REPLACE

        RN = self.args.k_loop
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        query_nums = []
        arm_preds = [pred_g]
        final_emb = data
        for n in range(RN):
            start_random = time.time()
            K_set = random.sample(allCode, N_REP)
            arm_chain = []
            arm_pred = []
            iteration = 0
            N = np.ones(len(K_set))
            time_Dur = 0
            robust_flag = 1
            new_words = copy.deepcopy(sentences)
            new_data = copy.deepcopy(data)
            index_candidate, pred_set_list, data_col = self.word_paraphrase_nocombine(new_words, K_set, length,
                                                                                      list_closest_neighbors, y,
                                                                                      new_data)
            INDEX = []
            while robust_flag == 1 and len(Counter(arm_chain).keys()) <= Budget * num_sent and time_Dur <= SECONDS:
                iteration += 1

                ucb = self.UCB(iteration, pred_set_list, N)
                topk_feature_index = np.argsort(ucb)[-1]
                INDEX.append(topk_feature_index)

                Feat_max = K_set[topk_feature_index]
                cand_max, pred_max, data_col = self.word_paraphrase_nocombine(new_words, [Feat_max], length,
                                                                              list_closest_neighbors, y, new_data)
                final_emb = data_col[0]
                new_data = torch.FloatTensor(data_col[0])
                new_words = self.New_Word(new_words, [(Feat_max, cand_max[0])])

                arm_chain.append((Feat_max, cand_max[0]))
                arm_pred.append(pred_max)

                n_add = np.eye(len(N))[topk_feature_index]
                N += n_add

                pred_set_list_add = np.zeros(len(K_set))
                pred_set_list_add[topk_feature_index] = pred_max
                pred_set_list = pred_set_list + pred_set_list_add

                time_end = time.time()
                time_Dur = time_end - start_random
                if arm_pred[-1] > 0:
                    success.append(1)
                    num_armchain.append(len(arm_chain))
                    n_change.append(len(Counter(arm_chain).keys()))
                    time_success.append(time_Dur)
                    arm_chains.append(arm_chain)
                    arm_preds.append(arm_pred)
                    query_nums.append(iteration * (N_REP * 10 + 1))
                    robust_flag = 0
                    break
                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break
            if time_Dur > SECONDS:
                print('The time is over', time_Dur, file=log_f, flush=True)
                break

        return success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change, query_nums, NoAttack, num_sent, \
               final_emb


def main():
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    Dataset = args.dataset
    SECONDS = args.time
    Budget = args.budget
    Data_Type = feature = args.feature
    model_name = args.model
    N_REPLACE = args.nrep
    alpha = args.alpha
    loop = args.k_loop
    text_embs = pickle.load(open('./politifact/' + feature + '_' + model_name + '_emb.pickle', 'rb'))
    text_labels = pickle.load(open('./politifact/labels.pickle', 'rb'))
    print(args)
    texts = pickle.load(open('./politifact/texts.pickle', 'rb'))
    test_idx = pickle.load(open('./politifact/test_idx.pickle', 'rb'))
    test_set = Pol_Dataset(text_embs, text_labels, args, 'test')
    args.num_classes = test_set.num_classes
    args.num_features = test_set.num_features
    texts_sentence_neighbors = pickle.load(open('./politifact/text_neighbors_scpn.pickle', 'rb'))
    sentences_all = pickle.load(open('./politifact/scpn_sentences.pickle', 'rb'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print(Dataset, Data_Type, model_name)

    file_dir = './Logs/' + Dataset + '/' + feature + '/' + model_name + '/ucb_base/'
    make_dir(file_dir)
    log_f = open(
        file_dir + 'scpn_Budget_%s_ALPHA=%s_N_REPLACE=%s_Time=%s_Loop_%s.bak' % (
            str(Budget), str(alpha), str(N_REPLACE), str(SECONDS), str(loop)), 'w+')

    TITLE = '===scpn_ ' + 'UCB_R:' + ' target_mf = ' + str(0) + ' changeRange = ' + str(N_REPLACE) + ' Time' + str(
        SECONDS) + ' Loop=' + str(loop) + ' ==='

    best_parameters_file = './classifier/politifact_' + feature + '_' + model_name + '.param'

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    attacker = Attacker(args, best_parameters_file)

    success_num = 0
    NoAttack_num = 0
    Total_iteration = 0
    Total_change = 0
    Total_time = 0
    Total_query_num = 0
    Total_flip_ratio = 0
    Total_embs = []

    count = -1
    for xs, ys, lengths in test_loader:
        count += 1
        print("====Sample %d/%d ======", count + 1, len(test_idx), file=log_f, flush=True)
        print(count)
        text = texts[test_idx[count]]
        label = ys[0]
        neighbors = texts_sentence_neighbors[count]
        sentences = sentences_all[test_idx[count]]
        success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change, query_nums, NoAttack_num, num_sent, \
        final_emb = attacker.attack(count, xs[0], text, label, NoAttack_num, log_f, N_REPLACE, neighbors, sentences,
                                    lengths[0])

        Total_embs.append(final_emb)
        if np.sum(success) >= 1:
            SR_sample = np.sum(success) / RN
            success_num += 1
            AI_sample = np.average(num_armchain)
            Achange = np.average(n_change)
            AT_sample = np.average(time_success)
            query_num = np.average(query_nums)
            flip_ratio = Achange / num_sent

            Total_iteration += AI_sample
            Total_change += Achange
            Total_time += AT_sample
            Total_query_num += query_num
            Total_flip_ratio += flip_ratio

            print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
            print(" Time: ", AT_sample, file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
        print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (count - NoAttack_num) != 0 and success_num != 0:
            print("--- success Ratio: " + str(success_num / ((count + 1) - NoAttack_num)) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Change: " + str(Total_change / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Query Num: " + str(Total_query_num / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Flip Ratio: " + str(np.around(Total_flip_ratio / success_num, decimals=3)) + " ---",
                  file=log_f, flush=True)
    pickle.dump(Total_embs, open(file_dir + 'final_embs_%.1f.pickle' % Budget, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    if args.feature == 'bert':
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        bert = BertModel.from_pretrained("bert-base-cased")
    main()
