import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import numpy as np
import pickle as pkl

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from bert_serving.client import BertClient
client = BertClient()

vectors = client.encode(['dog'])
vectors = client.encode(['First do it','then do it right', 'then do it better']) # vector 3* 728

from Model.lstm_bert import LSTMClassifier
device = -1
if torch.cuda.is_available():
    device = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', default='./Dataset/BinaryClass/train.csv',type=str,dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', default='./Dataset/BinaryClass/test_1.csv',type=str,dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--log-every', type=int, default=10000, help='Steps for each logging.')

    parser.add_argument('--batch-size', action='store', default=1, type=int,
                        help='Mini batch size.')
    parser.add_argument('--lr', action='store', default= 0.0001, type=float,
                        help='learning rate.')

    return parser.parse_args()

def evaluate(model, batch_text,yhat_classes,yhat_probs):

    preds =model(batch_text[0].cuda().unsqueeze(dim = 1))

    probs, classes = torch.exp(preds).max(dim=1)
    probs = probs.data.cpu().numpy().tolist()
    classes = classes.data.cpu().numpy().tolist()

    yhat_probs += probs
    yhat_classes += classes

    # eval_acc=sum([1 if np.exp(preds.data.cpu().numpy()[i][j])>0.5 else 0 for i,j in enumerate(batch.label.data.cpu().numpy()[0])]) # decide the classify result is right(1) or wrong (0)
    return preds,yhat_probs,yhat_classes

import csv
def re_order(in_file, out_file):
    with open(in_file, 'r', newline='') as in_file_handle:
        reader = csv.reader(in_file_handle)
        columns = [1, 0]
        content = []
        for row in reader:
            content.append([row[1],row[0]])
        with open(out_file, 'w', newline='') as out_file_handle:
            writer = csv.writer(out_file_handle,delimiter='\t')
            for i in content:
                writer.writerow(i)
            # writer.writerow(content)


def main():

    opt = parse_args()
    log_f = open('./Logs/BertLSTM/lr_%f_b=%d.bak'%(opt.lr,opt.batch_size), 'w+')
    TITLE = '===== ' + 'train_learning Rate'+ str(opt.lr)+' _ Batch Size' +str(opt.batch_size) +' ====='
    print(TITLE)
    print(TITLE,file=log_f, flush=True)
    src_field = data.Field()
    label_field = data.Field(pad_token=None, unk_token=None)
    import pandas as pd
    #'this is for creating proper data source '
    # re_order('.data/yelp_review_full_csv/train.csv', './data/YelpFull/train.tsv')

    train = data.TabularDataset(
        path=opt.train_path, format='csv',
        fields=[('text', src_field), ('label', label_field)]
    )

    test = data.TabularDataset(
        path=opt.test_path, format='csv',
        fields=[('text', src_field), ('label', label_field)]
    )


    train_embed = []
    train_label = []
    for k in range(len(train)):#len(train)
        sample = [[i] for i in train[k].text ]
        train_embed.append(torch.from_numpy(client.encode(sample,is_tokenized=True)))
        train_label.append(int(0) if train[k].label == ["FALSE"] else int(1))

    train_label = torch.LongTensor(train_label)


    test_embed = []
    test_label = []
    for k in range(len(test)):
        sample = [[i] for i in test[k].text]
        test_embed.append(torch.from_numpy(client.encode(sample,is_tokenized=True)))
        test_label.append(int(0) if test[k].label == ["FALSE"] else int(1))
    test_label = torch.LongTensor(test_label)


    print("Training size: {0}, Testing size: {1}".format(len(train), len(test)),file=log_f, flush=True)

    classifier = LSTMClassifier(768, 512,2)

    if torch.cuda.is_available():
        classifier.cuda()
        for param in classifier.parameters():
            param.data.uniform_(-0.08, 0.08)

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(classifier.parameters(),lr=opt.lr)

    step = 0
    for epoch in range(0,500):
        running_loss = 0.0
        train_acc = 0
        trainy = []
        trainyhat_classes = []
        trainyhat_probs = []
        print('================== This is the %d epoch===================== '% (epoch),file=log_f, flush=True)
        classifier.train()
        batch_num = len(train_embed)//opt.batch_size
        for batch_idx in range(batch_num):
            batch_text = train_embed[batch_idx*opt.batch_size: (batch_idx+1)*opt.batch_size]
            batch_label = train_label[batch_idx*opt.batch_size: (batch_idx+1)*opt.batch_size]
            trainy = trainy + list(batch_label.numpy())

            optimizer.zero_grad()
            preds, trainyhat_probs, trainyhat_classes = evaluate(classifier, batch_text, trainyhat_classes,
                                                                 trainyhat_probs)

            loss = criterion(preds, (batch_label.view(-1)).cuda())
            running_loss += loss.data
            loss.backward()
            optimizer.step()
            step += 1
            if step % opt.log_every == 0:
                print('For the epoch step [%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / opt.log_every),file=log_f, flush=True)
                running_loss = 0.0

        print('======This is the TRAINing part====== ', file=log_f, flush=True)
        trainy = np.array(trainy)
        trainyhat_classes = np.array(trainyhat_classes)
        trainyhat_probs = np.array(trainyhat_probs)
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(trainy, trainyhat_classes)
        print('Accuracy: %f' % accuracy, file=log_f, flush=True)
        # precision tp / (tp + fp)
        precision = precision_score(trainy, trainyhat_classes)
        print('Precision: %f' % precision, file=log_f, flush=True)
        # recall: tp / (tp + fn)
        recall = recall_score(trainy, trainyhat_classes)
        print('Recall: %f' % recall, file=log_f, flush=True)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(trainy, trainyhat_classes)
        print('F1 score: %f' % f1, file=log_f, flush=True)

        # ROC AUC
        auc = roc_auc_score(trainy, trainyhat_probs)
        print('ROC AUC: %f' % auc, file=log_f, flush=True)

        torch.save(classifier, os.path.join('./Model/BertLSTM/lr_%f_b=%d_LSTM_{%d}.bak'%(opt.lr,opt.batch_size,epoch+1)))

        # classifier = torch.load('./Model/LSTM_log/lr_0.000010_b=1_LSTM_{87}.bak', map_location=lambda storage, loc: storage).cuda()

        print('======This is the TESTing part====== ',file=log_f, flush=True)
        classifier.eval()
        testy = []
        yhat_classes = []
        yhat_probs = []
        batch_num = len(test_embed) // opt.batch_size
        for batch_idx in range(batch_num):
            batch_text = test_embed[batch_idx * opt.batch_size: (batch_idx + 1) * opt.batch_size]
            batch_label = test_label[batch_idx * opt.batch_size: (batch_idx + 1) * opt.batch_size]

            preds, yhat_probs, yhat_classes = evaluate(classifier, batch_text, yhat_classes,
                                                                 yhat_probs)

            testy = testy + list(batch_label.numpy())


        testy = np.array(testy)

        yhat_classes=np.array(yhat_classes)
        yhat_probs = np.array(yhat_probs)
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(testy, yhat_classes)
        print('Accuracy: %f' % accuracy,file=log_f, flush=True)
        # precision tp / (tp + fp)
        precision = precision_score(testy, yhat_classes)
        print('Precision: %f' % precision,file=log_f, flush=True)
        # recall: tp / (tp + fn)
        recall = recall_score(testy, yhat_classes)
        print('Recall: %f' % recall,file=log_f, flush=True)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(testy, yhat_classes)
        print('F1 score: %f' % f1,file=log_f, flush=True)
        CM = confusion_matrix(testy, yhat_classes)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        print("TN,FN,TP,FP",TN,FN,TP,FP,file=log_f, flush=True)
        # False negative rate
        FNR = FN / (TP + FN)
        # False Positive rate
        FPR = FP/(FP+TN)

        print("FPR,FNR",FPR,FNR, file=log_f, flush=True)
        # ROC AUC
        auc = roc_auc_score(testy, yhat_probs)
        print('ROC AUC: %f' % auc,file=log_f, flush=True)



if __name__ == '__main__':
    main()
