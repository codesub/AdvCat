import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from util import *


from Model.lstm import LSTMClassifier
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
    parser.add_argument('--lr', action='store', default= 0.00001, type=float,
                        help='learning rate.')

    return parser.parse_args()

def evaluate(model, batch,yhat_classes,yhat_probs):
    inputs=batch.text #F.pad(batch.text.transpose(0,1), (0,sentence_len-len(batch.text)))
    preds =model(inputs.cuda())

    probs, classes = torch.exp(preds).max(dim=1)
    probs = probs.data.cpu().numpy().tolist()
    classes = classes.data.cpu().numpy().tolist()

    yhat_probs += probs
    yhat_classes += classes

    # eval_acc=sum([1 if np.exp(preds.data.cpu().numpy()[i][j])>0.5 else 0 for i,j in enumerate(batch.label.data.cpu().numpy()[0])]) # decide the classify result is right(1) or wrong (0)
    return preds,yhat_probs,yhat_classes

#

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
    log_f = open('./Logs/W2VLSTM/lr_%f_b=%d.bak'%(opt.lr,opt.batch_size), 'w+')
    TITLE = '===== ' + 'train_learning Rate'+ str(opt.lr)+' _ Batch Size' +str(opt.batch_size) +' ====='
    print(TITLE)
    print(TITLE,file=log_f, flush=True)
    src_field = data.Field()
    label_field = data.Field(pad_token=None, unk_token=None)


    train = data.TabularDataset(
        path=opt.train_path, format='csv',
        fields=[('text', src_field), ('label', label_field)]
    )

    test = data.TabularDataset(
        path=opt.test_path, format='csv',
        fields=[('text', src_field), ('label', label_field)]
    )
    src_field.build_vocab(train, max_size=100000, min_freq=2, vectors="glove.6B.300d")
    label_field.build_vocab(train)

    print("Training size: {0}, Testing size: {1}".format(len(train), len(test)),file=log_f, flush=True)

    classifier = LSTMClassifier(300, 512, len(label_field.vocab), src_field.vocab.vectors)

    train_iter = data.BucketIterator(
        dataset=train,
        batch_size=opt.batch_size,
        device=device,
        repeat=False
    )
    test_iter = data.BucketIterator(
        dataset=test,
        batch_size=5,
        device=device,
        repeat=False
    )


    if torch.cuda.is_available():
        classifier.cuda()
        for param in classifier.parameters():
            param.data.uniform_(-0.08, 0.08)

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(classifier.parameters(),lr=opt.lr)

    # step = 0
    for epoch in range(0,1000):
        step = 0
        running_loss = 0.0
        train_acc = 0
        trainy = []
        trainyhat_classes = []
        trainyhat_probs = []
        print('================== This is the %d epoch===================== '% (epoch),file=log_f, flush=True)
        classifier.train()
        for batch in train_iter:
            optimizer.zero_grad()
            preds, trainyhat_probs, trainyhat_classes = evaluate(classifier, batch, trainyhat_classes, trainyhat_probs)

            trainy = trainy + list(batch.label.numpy()[0])

            loss = criterion(preds, (batch.label.view(-1)).cuda())
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

        torch.save(classifier, os.path.join('./Model/W2VLSTM/lr_%f_b=%d_LSTM_{%d}.bak'%(opt.lr,opt.batch_size,epoch)))

        print('======This is the TESTing part====== ',file=log_f, flush=True)
        classifier.eval()
        testy = []
        yhat_classes = []
        yhat_probs = []
        for batch in test_iter:
            testy = testy + list(batch.label.numpy()[0])
            preds,yhat_probs,yhat_classes = evaluate(classifier, batch,yhat_classes,yhat_probs)

        testy = np.array(testy)
        yhat_classes =np.array(yhat_classes)
        yhat_probs=np.array(yhat_probs)
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
