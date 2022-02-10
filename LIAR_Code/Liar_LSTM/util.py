import csv
from torchtext import data
import torch
from torch.autograd import Variable
from math import exp
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def read_data(path, pos):
    x, y = [], []
    count=0
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            count+=1
            if count>20000: break
            x.append(row[0][:])#.decode('utf-8'))
            y.append(1 if row[1]==pos else 0)
    return x, y

def read_data_multi_class(path):
    x, y = [], []
    count=0
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            count+=1
            if count>20000: break
            row = row[0].split('.,'or '.",')
            if len(row) == 2:
                x.append(row[0])  # .decode('utf-8'))
                if row[1] == 'TRUE':
                    y.append(int(1))
                else:
                    y.append(int(0))
    return x, y

def read_data_multi_class_test(path):
    x, y = [], []
    count=0
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            count+=1
            if count>100: break
            x.append(row[0][:])#.decode('utf-8'))
            y.append(int(row[1])-1)
    return x, y

def dump_data(path, x, y):
    with open(path, 'w') as fout:
        writer = csv.writer(fout, delimiter="\t")
        for doc, label in zip(x, y):
            writer.writerow([u" ".join(doc), label])

def dump_row(path, doc, label):
    with open(path, 'a') as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow([u" ".join(doc), label])

def v2l(v):
    return 'FAKE' if v else 'REAL'

def dump_p(path, p):
    with open(path, 'w') as fout:
        writer = csv.writer(fout,delimiter="\t")
        for count, doc, pred, pred_prob, changed_pos in p:
            writer.writerow([count, u" ".join(doc), v2l(pred), pred_prob, ",".join([str(i) for i in changed_pos])])

def dump_p_row(path, p):
    count, doc, pred, pred_prob, changed_pos=p[0], p[1], p[2], p[3], p[4]
    with open(path, 'a') as fout:
        writer = csv.writer(fout,delimiter="\t")
        writer.writerow([count, u" ".join(doc), v2l(pred), pred_prob, ",".join([str(i) for i in changed_pos])])

def text_to_var(docs, vocab):
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

def classify(var, model):
    if 'CNN' in str(type(model)):
        pred,_ = model(var)
        prob, clazz = pred[0].max(dim=0)
        return prob.data[0], clazz.data[0]
    else:
        pred = model(var)
        prob, clazz = pred[0].max(dim=0)
        return exp(prob), clazz

def classifybert(sample_text, model):
    preds = model(sample_text.cuda().unsqueeze(dim=1))
    prob, clazz = preds[0].max(dim=0)
    # orig_pred = (np.argmax(np.exp(preds.data.cpu().numpy()), axis=1))
    # orig_prob = (np.max(np.exp(preds.data.cpu().numpy()), axis=1))

    return exp(prob), clazz
def negative_score(var, model, y):
    if 'CNN' in str(type(model)):
        pred_probs,_ = model(var)
        c_prob = pred_probs[:, 1-y].data[0]
    else:
        pred_probs = model(var)
        c_prob = exp(pred_probs[:, 1-y].data[0])
    return c_prob


def performance(testy,yhat_classes,yhat_probs,log_f):

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy, file=log_f, flush=True)
    # precision tp / (tp + fp)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision, file=log_f, flush=True)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall, file=log_f, flush=True)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1, file=log_f, flush=True)
    CM = confusion_matrix(testy, yhat_classes)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TN,FN,TP,FP", TN, FN, TP, FP, file=log_f, flush=True)

    # False negative rate
    FNR = FN / (TP + FN)
    # False Positive rate
    FPR = FP / (FP + TN)

    print("FPR,FNR", FPR, FNR, file=log_f, flush=True)
    # ROC AUC
    auc = roc_auc_score(testy, yhat_probs)
    print('ROC AUC: %f' % auc, file=log_f, flush=True)

def SuccessPosValue(a,pos):
    b = []
    for i in pos:
        b.append(a[i[0]])

    return b