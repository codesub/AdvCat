import argparse
from util import *
from models import *
import time
from tqdm import tqdm


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


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=10000, help='maximum number of epochs')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [spacy, bert]')
parser.add_argument('--model', type=str, default='lstm', help='model type, [mlp, lstm]')
parser.add_argument('--data_type', type=str, default='clean', help='data type, [clean, perturbed]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

model_name = args.model
feature = args.feature

text_embs = pickle.load(open('./politifact/' + feature + '_' + model_name + '_emb.pickle', 'rb'))
text_labels = pickle.load(open('./politifact/labels.pickle', 'rb'))

print(args)

training_set = Pol_Dataset(text_embs, text_labels, args, 'train')
test_set = Pol_Dataset(text_embs, text_labels, args, 'test')
validation_set = Pol_Dataset(text_embs, text_labels, args, 'val')
args.num_classes = training_set.num_classes
args.num_features = training_set.num_features
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
if model_name == 'mlp':
    model = MLP(args)
else:
    model = LSTM(args)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if __name__ == '__main__':
    # Model training
    min_loss = 1e10
    val_loss_values = []
    t = time.time()

    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        model.train()
        for x, y, length in train_loader:
            optimizer.zero_grad()
            x = x.to(args.device)
            y = y.to(args.device)
            if model_name == 'mlp':
                out = model(x)
            else:
                out = model(x, length)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

    _, loss_train = compute_test(train_loader)
    print('Final training loss:', loss_train)
    [acc, f1_macro, f1_micro, precision, recall, FNR, FPR], test_loss = compute_test(test_loader, verbose=False)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}, FNR: {FNR:.4f}, FPR: {FPR:.4f}')
    print(args)
    torch.save(model.state_dict(), 'classifier/' + args.dataset + '_' + args.feature + '_' + args.model + '.param',
               _use_new_zipfile_serialization=False)
