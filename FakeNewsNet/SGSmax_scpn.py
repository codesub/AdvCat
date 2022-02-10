import time
from itertools import combinations
import argparse
import spacy
from models import *
from util import *

parser = argparse.ArgumentParser(description='FSGS')
parser.add_argument('--budget', default=0.4, type=float, help='purturb budget')
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--time', default=1800, type=int, help='time limit')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [spacy, bert]')
parser.add_argument('--model', type=str, default='mlp', help='model type, [mlp, lstm]')
parser.add_argument('--data_type', type=str, default='clean', help='data type, [clean, perturbed]')
parser.add_argument('--rand_k', default=10, type=int, help='randomly chose k features to iterate')
args = parser.parse_args()


class Attacker(object):
    def __init__(self, args, log_f):
        self.n_labels = args.num_classes
        if model_name == 'mlp':
            self.model = MLP(args)
        else:
            self.model = LSTM(args)
        self.model = self.model.to(args.device)
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.model.eval()
        self.log_f = log_f
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

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

    def eval_object(self, eval_funccall, x, current_object, greedy_set, orig_label, pos, query_num, num_sent, length):
        best_temp_funccall = copy.deepcopy(eval_funccall)
        candidate_lists = []
        success_flag = 1
        funccall_lists = []
        data_lists = torch.tensor([])
        change_flag = 0
        worst_object = current_object
        emb = x
        if self.args.model == 'mlp':
            text = ' '.join(eval_funccall)
            if self.args.feature == 'spacy':
                doc_vec = nlp(text).vector
                data = torch.from_numpy(doc_vec)
                data = data.to(self.args.device)
            else:
                encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
                embedding = bert(**encoded_input)
                data = embedding[1][0].data
                data = data.to(self.args.device)
        else:
            data = copy.deepcopy(x)
            if self.args.feature == 'spacy':
                doc_vec = nlp(pos[1]).vector
                data[pos[0]] = torch.from_numpy(doc_vec)
                data = data.to(self.args.device)
            else:
                encoded_input = tokenizer(pos[1], return_tensors='pt', max_length=512, truncation=True)
                embedding = bert(**encoded_input)
                data[pos[0]] = embedding[1][0].data
                data = data.to(self.args.device)
        eval_pred, eval_g, eval_h = self.classify(data, y, length)
        query_num += 1
        object = eval_h - eval_g
        if object > 0:
            emb = data.cpu()
            return object, eval_funccall, 0, query_num, emb
        if object >= worst_object:
            emb = data.cpu()
            change_flag = 1
            worst_object = object
        if greedy_set:
            for i in range(1, min(len(greedy_set) + 1, int(np.ceil(budget * num_sent)))):
                subset1 = combinations(greedy_set, i)
                for subset in subset1:
                    candidate_lists.append(list(subset))

        for can in candidate_lists:

            temp_funccall = copy.deepcopy(eval_funccall)
            temp_data = copy.deepcopy(data)
            temp_data = temp_data.cpu()

            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx
            if self.args.model == 'mlp':
                if self.args.feature == 'spacy':
                    text = ' '.join(temp_funccall)
                    doc_vec = nlp(text).vector
                    temp_data = torch.from_numpy(doc_vec)
                else:
                    text = ' '.join(temp_funccall)
                    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
                    embedding = bert(**encoded_input)
                    temp_data = embedding[1][0].data
            else:
                if self.args.feature == 'spacy':
                    for position in can:
                        visit_idx = position[0]
                        code_idx = position[1]
                        temp_funccall[visit_idx] = code_idx
                        doc_vec = nlp(code_idx).vector
                        temp_data[visit_idx] = torch.from_numpy(doc_vec)
                else:
                    for position in can:
                        visit_idx = position[0]
                        code_idx = position[1]
                        temp_funccall[visit_idx] = code_idx
                        encoded_input = tokenizer(code_idx, return_tensors='pt', max_length=512, truncation=True)
                        embedding = bert(**encoded_input)
                        temp_data[visit_idx] = embedding[1][0].data
            temp_data = temp_data.unsqueeze(0)

            funccall_lists.append(temp_funccall)
            data_lists = torch.cat((data_lists, temp_data), dim=0)

        query_num += len(funccall_lists)
        batch_size = self.args.batch_size
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))
        for i in range(n_batches):
            data_temp = data_lists[batch_size * i: batch_size * (i + 1)].to(self.args.device)
            if self.args.model == 'mlp':
                logit = self.model(data_temp)
            else:
                logit = self.model(data_temp, [length]*len(data_temp))
            logit = F.softmax(logit, dim=1)
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]
            subsets_h = 1 - subsets_g
            subsets_object = subsets_h - subsets_g
            max_object = np.max(subsets_object)
            max_index = np.argmax(subsets_object)

            if max_object >= worst_object:
                change_flag = 1
                worst_object = max_object
                best_temp_funccall = copy.deepcopy(funccall_lists[batch_size * i + max_index])
                emb = copy.deepcopy(data_lists[batch_size * i + max_index])

        if change_flag == 0:
            success_flag = 2

        if worst_object > 0:
            success_flag = 0

        return worst_object, best_temp_funccall, success_flag, query_num, emb

    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set()
        for i in range(len(eval_funccall)):
            if eval_funccall[i] != new_funccall[i]:
                diff_set.add(i)
        return diff_set

    def attack(self, data, y, length, sentence_neighbors, sentences):
        print()
        st = time.time()
        success_flag = 1
        orig_pred, orig_g, orig_h = self.classify(data, y, length)

        num_sent = len(sentences)
        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = sentences
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(float(orig_g))
        mf_process.append(float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0
        query_num = 0
        final_emb = data

        current_object = orig_h - orig_g
        flip_funccall = sentences
        list_neighbors = sentence_neighbors
        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                   query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                   greedy_set_best_temp_funccall, \
                   n_changed, flip_funccall, flip_set, iteration, num_sent, final_emb

        print(current_object)

        while success_flag == 1:
            iteration += 1
            success_flag = 1
            candidate_objects = []
            candidate_funccalls = []
            candidate_poses = []
            candidate_embs = []
            avail_fea = set(range(min(len(sentences), 50))) - greedy_set_visit_idx
            random_k = args.rand_k
            visit_list = np.random.choice(list(avail_fea), min(random_k, len(avail_fea)), replace=False)

            for visit_idx in visit_list:
                if visit_idx in greedy_set_visit_idx:
                    continue
                worst_object_cate = -2
                best_pos_cate = -1
                best_temp_funccall_cate = copy.deepcopy(sentences)
                temp_emb = copy.deepcopy(final_emb)
                for code_idx in list_neighbors[visit_idx]:

                    pos = (visit_idx, code_idx)

                    eval_funccall = copy.deepcopy(sentences)
                    eval_funccall[visit_idx] = code_idx
                    worst_object, temp_funccall, success_flag_temp, query_num, emb = self.eval_object(eval_funccall, data,
                                                                                                 current_object,
                                                                                                 greedy_set, y, pos,
                                                                                                 query_num, num_sent,
                                                                                                 length)

                    if success_flag_temp == 2:
                        temp_funccall = greedy_set_best_temp_funccall

                    if success_flag_temp == 0:
                        success_flag = 0

                    if worst_object > worst_object_cate:
                        worst_object_cate = worst_object
                        best_pos_cate = pos
                        best_temp_funccall_cate = copy.deepcopy(temp_funccall)
                        temp_emb = copy.deepcopy(emb)

                candidate_objects.append(worst_object_cate)
                candidate_funccalls.append(best_temp_funccall_cate)
                candidate_poses.append(best_pos_cate)
                candidate_embs.append(temp_emb)

            if len(greedy_set) == len(sentences):
                robust_flag = 1
                print('Greedily chose all features already', file=self.log_f, flush=True)
                break
            index = np.argmax(candidate_objects)
            max_object = np.max(candidate_objects)
            if max_object == -2:
                robust_flag = 1
                print('Greedily chose all valid features already', file=self.log_f, flush=True)
                break
            max_pos = candidate_poses[index]
            greedy_set_best_temp_funccall = candidate_funccalls[index]
            final_emb = candidate_embs[index]

            print(iteration)
            print('query', query_num)
            print(max_object)

            greedy_set.add(max_pos)
            greedy_set_visit_idx.add(max_pos[0])
            g_process.append((1-max_object.item())/2)
            mf_process.append(max_object.item())
            greedy_set_process.append(copy.deepcopy(greedy_set))
            if max_object > current_object:
                current_object = max_object
            changed_set_process.append(self.changed_set(sentences, greedy_set_best_temp_funccall))

            print(greedy_set)

            if success_flag == 1:
                if (time.time() - st) > time_limit:
                    success_flag = -1
                    robust_flag = 1

        n_changed = len(self.changed_set(sentences, greedy_set_best_temp_funccall))
        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(sentences, flip_funccall)
            print('attack successful')

        print("Modified_set:", flip_set)
        print(flip_funccall)

        return g_process, mf_process, greedy_set_process, changed_set_process, \
               query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
               greedy_set_best_temp_funccall, \
               n_changed, flip_funccall, flip_set, iteration, num_sent, final_emb


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
Dataset = args.dataset
Model_Type = feature = args.feature
budget = args.budget
time_limit = args.time
model_name = args.model
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
print(Dataset, Model_Type, model_name)
output_file = './Logs/%s/%s/%s/%s/' % (Dataset, Model_Type, model_name, 'SGS')
nlp = spacy.load("en_core_web_lg")
if args.feature == 'bert':
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert = BertModel.from_pretrained("bert-base-cased")
if os.path.isdir(output_file):
    pass
else:
    os.mkdir(output_file)

best_parameters_file = './classifier/politifact_' + feature + '_' + model_name + '.param'

g_process_all = []
mf_process_all = []
greedy_set_process_all = []
changed_set_process_all = []

query_num_all = []
robust_flag_all = []

orignal_funccalls_all = []
orignal_labels_all = []

final_greedy_set_all = []
final_greedy_set_visit_idx_all = []
final_changed_num_all = []
final_funccall_all = []
final_embs_all = []

flip_funccall_all = []
flip_set_all = []
flip_mf_all = []
flip_sample_original_label_all = []
flip_sample_index_all = []
flip_percent_all = []

iteration_all = []
time_all = []

log_attack = open(output_file + 'greedmax_Attack_%.1f.bak' % budget, 'w+')
attacker = Attacker(args, log_attack)

i = -1
for x, y, lengths in test_loader:
    i += 1
    print(i)
    print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

    text = texts[test_idx[i]]
    label = y[0]
    neighbors = texts_sentence_neighbors[i]
    sentences = sentences_all[test_idx[i]]

    print('* Processing:%d/%d person' % (i, len(test_loader)), file=log_attack, flush=True)

    try:
        print("* Original: " + str(text), file=log_attack, flush=True)
    except UnicodeEncodeError:
        pass

    print("  Original label: %d" % label, file=log_attack, flush=True)

    st = time.time()
    g_process, mf_process, greedy_set_process, changed_set_process, query_num, robust_flag, \
    greedy_set, greedy_set_visit_idx, greedy_set_best_temp_funccall, \
    num_changed, flip_funccall, flip_set, iteration, num_sent, final_emb = attacker.attack(x[0], label, lengths[0],
                                                                                           neighbors, sentences)
    print("Orig_Prob = " + str(g_process[0]), file=log_attack, flush=True)
    if robust_flag == -1:
        print('Original Classification Error', file=log_attack, flush=True)
    else:
        print("* Result: ", file=log_attack, flush=True)
    et = time.time()
    all_t = et - st


    if robust_flag == 1:
        print("This sample is robust.", file=log_attack, flush=True)

    if robust_flag != -1:
        print('g_process:', g_process, file=log_attack, flush=True)
        print('mf_process:', mf_process, file=log_attack, flush=True)
        print('greedy_set_process:', greedy_set_process, file=log_attack, flush=True)
        print('changed_set_process:', changed_set_process, file=log_attack, flush=True)
        print("  Number of query for this: " + str(query_num), file=log_attack, flush=True)
        print('greedy_set: ', file=log_attack, flush=True)
        print(greedy_set, file=log_attack, flush=True)
        print('greedy_set_visit_idx: ', file=log_attack, flush=True)
        print(greedy_set_visit_idx, file=log_attack, flush=True)
        print('greedy_funccall:', file=log_attack, flush=True)
        print(greedy_set_best_temp_funccall, file=log_attack, flush=True)
        print('best_prob = ' + str(g_process[-1]), file=log_attack, flush=True)
        print('best_object = ' + str(mf_process[-1]), file=log_attack, flush=True)
        print("  Number of changed codes: %d" % num_changed, file=log_attack, flush=True)
        print("risk funccall:", file=log_attack, flush=True)
        print('iteration: ' + str(iteration), file=log_attack, flush=True)
        print(" Time: " + str(all_t), file=log_attack, flush=True)
        if robust_flag == 0:
            print('flip_funccall:', file=log_attack, flush=True)
            print(flip_funccall, file=log_attack, flush=True)
            print('flip_set:', file=log_attack, flush=True)
            print(flip_set, file=log_attack, flush=True)
            print('flip_percent:', num_changed/num_sent, file=log_attack, flush=True)
            print('flip_object = ', mf_process[-1], file=log_attack, flush=True)
            print(" The cardinality of S: " + str(len(greedy_set)), file=log_attack, flush=True)
        else:
            print(" The cardinality of S: " + str(len(greedy_set)) + ', but timeout', file=log_attack,
                  flush=True)

    time_all.append(all_t)
    g_process_all.append(copy.deepcopy(g_process))
    mf_process_all.append(copy.deepcopy(mf_process))
    greedy_set_process_all.append(copy.deepcopy(greedy_set_process))
    changed_set_process_all.append(copy.deepcopy(changed_set_process))

    query_num_all.append(query_num)
    robust_flag_all.append(robust_flag)
    iteration_all.append(iteration)

    orignal_funccalls_all.append(copy.deepcopy(sentences))
    orignal_labels_all.append(label)

    final_greedy_set_all.append(copy.deepcopy(greedy_set))
    final_greedy_set_visit_idx_all.append(copy.deepcopy(greedy_set_visit_idx))
    final_funccall_all.append(copy.deepcopy(greedy_set_best_temp_funccall))
    final_changed_num_all.append(num_changed)
    final_embs_all.append(final_emb.tolist())

    if robust_flag == 0:
        flip_funccall_all.append(copy.deepcopy(flip_funccall))
        flip_set_all.append(copy.deepcopy(flip_set))
        flip_mf_all.append(mf_process[-1])
        flip_sample_original_label_all.append(label)
        flip_sample_index_all.append(i)
        flip_percent_all.append(num_changed/num_sent)

pickle.dump(g_process_all,
            open(output_file + 'g_process_%.1f.pickle' % budget, 'wb'))
pickle.dump(mf_process_all,
            open(output_file + 'mf_process_%.1f.pickle' % budget, 'wb'))
pickle.dump(greedy_set_process_all,
            open(output_file + 'greedy_set_process_%.1f.pickle' % budget, 'wb'))
pickle.dump(changed_set_process_all,
            open(output_file + 'changed_set_process_%.1f.pickle' % budget, 'wb'))
pickle.dump(query_num_all,
            open(output_file + 'querynum_%.1f.pickle' % budget, 'wb'))
pickle.dump(robust_flag_all,
            open(output_file + 'robust_flag_%.1f.pickle' % budget, 'wb'))
pickle.dump(orignal_funccalls_all,
            open(output_file + 'original_funccall_%.1f.pickle' % budget, 'wb'))
pickle.dump(orignal_labels_all,
            open(output_file + 'original_label_%.1f.pickle' % budget, 'wb'))
pickle.dump(final_greedy_set_all,
            open(output_file + 'greedy_set_%.1f.pickle' % budget, 'wb'))
pickle.dump(final_greedy_set_visit_idx_all,
            open(output_file + 'feature_greedy_set_%.1f.pickle' % budget, 'wb'))
pickle.dump(final_changed_num_all,
            open(output_file + 'changed_num_%.1f.pickle' % budget, 'wb'))
pickle.dump(final_funccall_all,
            open(output_file + 'modified_funccall_%.1f.pickle' % budget, 'wb'))
pickle.dump(final_embs_all,
            open(output_file + 'final_embs_%.1f.pickle' % budget, 'wb'))
pickle.dump(flip_funccall_all,
            open(output_file + 'flip_funccall_%.1f.pickle' % budget, 'wb'))
pickle.dump(flip_set_all,
            open(output_file + 'flip_set_%.1f.pickle' % budget, 'wb'))
pickle.dump(flip_mf_all,
            open(output_file + 'flip_mf_%.1f.pickle' % budget, 'wb'))
pickle.dump(flip_sample_original_label_all,
            open(output_file + 'flip_sample_original_label_%.1f.pickle' % budget, 'wb'))
pickle.dump(flip_sample_index_all,
            open(output_file + 'flip_sample_index_%.1f.pickle' % budget, 'wb'))
pickle.dump(flip_percent_all,
            open(output_file + 'flip_percent_%.1f.pickle' % budget, 'wb'))
pickle.dump(iteration_all,
            open(output_file + 'iteration_%.1f.pickle' % budget, 'wb'))
pickle.dump(time_all,
            open(output_file + 'time_%.1f.pickle' % budget, 'wb'))

write_file(Dataset, Model_Type, model_name, budget, 'SGS', time_limit)
