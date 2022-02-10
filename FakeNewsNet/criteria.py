from util import *
from models import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [spacy, bert]')
parser.add_argument('--model', type=str, default='mlp', help='model type, [mlp, lstm]')
parser.add_argument('--data_type', type=str, default='perturbed', help='data type, [clean, perturbed]')
parser.add_argument('--budget', default=0.4, type=float, help='purturb budget')
parser.add_argument('--attack', default='SGS', type=str, help='attack algorithm')

args = parser.parse_args()
model_name = args.model
feature = args.feature
attack = args.attack
if args.data_type == 'clean':
    test_idx = pickle.load(open('./politifact/test_idx.pickle', 'rb'))
    text_embs = pickle.load(open('./politifact/' + feature + '_' + model_name + '_emb.pickle', 'rb'))
    text_labels = pickle.load(open('./politifact/labels.pickle', 'rb'))
    eval_dataset = Pol_Dataset(text_embs, text_labels, args, 'test')
else:
    text_embs = pickle.load(
        open('./Logs/politifact/' + feature + '/' + model_name + '/' + attack + '/final_embs_%.1f.pickle' % args.budget,
             'rb'))
    text_labels = pickle.load(open('./politifact/labels.pickle', 'rb'))
    eval_dataset = Pol_Dataset(text_embs, text_labels, args, 'eval')
args.num_classes = eval_dataset.num_classes
args.num_features = eval_dataset.num_features
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
model_file = './classifier/politifact_' + feature + '_' + model_name + '.param'
if model_name == 'mlp':
    model = MLP(args)
else:
    model = LSTM(args)
model.load_state_dict(torch.load(model_file, map_location='cpu'))
model = model.to(args.device)


@torch.no_grad()
def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    for x, y, length in loader:
        x = x.to(args.device)
        y = y.to(args.device)
        if model_name == 'mlp':
            out = model(x)
        else:
            out = model(x, length)
        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())
        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()
    return eval_deep(out_log, loader), loss_test


[acc, f1_macro, f1_micro, precision, recall, FNR, FPR, auc], test_loss = compute_test(eval_loader, verbose=False)
print(f'Test set results: acc: {acc:.4f}, '
      f'precision: {precision:.4f}, recall: {recall:.4f}, f1_macro: {f1_macro:.4f}, auc: {auc:.4f}, FNR: {FNR:.4f}, FPR: {FPR:.4f}')
