
import argparse
import os
import torch.optim as optim
import numpy as np
from util import *
import dataset
import model_CNN as model
import training
from bert_serving.client import BertClient
client = BertClient()
import matplotlib.pyplot as plt


def main():
    
    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TextCNN')
    #Training args

    parser.add_argument('--spacy-lang', type=str, default='en', 
                        help='language choice for spacy to tokenize the text')
    parser.add_argument('--pretrained', type=str, default='glove.6B.300d',
                    help='choice of pretrined word embedding from torchtext')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--batch-size', type=int, default=1,
                    help='input batch size for training (default: 64)')
    
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='input batch size for testing (default: 64)')
    
    parser.add_argument('--kernel-height', type=str, default='3,4,5',
                    help='how many kernel width for convolution (default: 3, 4, 5)')
    
    parser.add_argument('--out-channel', type=int, default=100,
                    help='output channel for convolutionaly layer (default: 100)')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate for linear layer (default: 0.5)')
    
    parser.add_argument('--num-class', type=int, default=2,
                        help='number of category to classify (default: 2)')
    
    #if you are using jupyternotebook with argparser
    args = parser.parse_known_args()[0]
    #args = parser.parse_args()

    log_f = open('./Logs/BertCNN/train_lr_%f_b=%d.bak' % (args.lr, args.batch_size), 'w+')
    TITLE = '===== ' + 'train_learning Rate' + str(args.lr) + ' _ Batch Size' + str(args.batch_size) + ' ====='
    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    
    #Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test, vocab = dataset.create_tabular_dataset('./train_1.csv',
                                 './test_1.csv',args.spacy_lang, args.pretrained)
    
    #%%Show some example to show the dataset
    print("Show some examples from train/valid..",file=log_f, flush=True)
    # print(trainset[0].text,  trainset[0].label)
    print(test[0].text,  test[0].label)
    train_embed = []
    train_label = []
    for k in range(len(train)):
        sample = [[i] for i in [x for x in train[k].text if x.strip()]]
        if len(sample) <= 10:
            sample = sample + (10 - len(sample)) * [sample[-1]]

        train_embed.append(torch.from_numpy(client.encode(sample, is_tokenized=True)))
        train_label.append(int(0) if train[k].label == 'FALSE' else int(1))

    train_label = torch.LongTensor(train_label)

    test_embed = []
    test_label = []
    for k in range(len(test)):
        sample = [[i] for i in [x for x in test[k].text if x.strip()]]
        if len(sample) <= 10:
            sample = sample + (10 - len(sample)) * [sample[-1]]

        test_embed.append(torch.from_numpy(client.encode(sample, is_tokenized=True)))
        test_label.append(int(0) if test[k].label == 'FALSE' else int(1))
    test_label = torch.LongTensor(test_label)

    print("Training size: {0}, Testing size: {1}".format(len(train), len(test)),file=log_f, flush=True)

    #%%Create
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = model.BertCNN(vocab,args.out_channel, kernels, args.dropout , args.num_class).to(device)
    # print the model summery
        
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_test_acc = -1
    
    #optimizer
    optimizer = optim.Adam(m.parameters(), lr=args.lr)
    step = 0
    for epoch in range(1, args.epochs+1):
        print('======This is the TRAINing part====== ', file=log_f, flush=True)
        #train loss
        trainy = []
        trainyhat_probs = []
        trainyhat_class = []

        batch_size = 1
        tr_loss, tr_acc,trainy,trainyhat_probs,trainyhat_class = training.trainbert(m, device, train_embed, optimizer, epoch, args.epochs,train_label,trainyhat_probs,batch_size,trainy,trainyhat_class)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc), file=log_f, flush=True)

        trainy = np.array(trainy)
        trainyhat_probs = np.array(trainyhat_probs)
        trainyhat_class = np.array(trainyhat_class)
        performance(trainy,trainyhat_class,trainyhat_probs,log_f)


        print('======This is the TESTing part====== ', file=log_f, flush=True)
        testy = []
        yhat_probs = []
        yhat_classes = []
        ts_loss, ts_acc,testy,yhat_probs,yhat_classes = training.validbert(m, device, test_embed,test_label,testy,yhat_probs,yhat_classes)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, ts_loss, ts_acc), file=log_f, flush=True)

        testy = np.array(testy)
        yhat_probs = np.array(yhat_probs)
        yhat_classes= np.array(yhat_classes)
        performance(testy,yhat_classes,yhat_probs,log_f)

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_test_acc))
            torch.save(m.state_dict(), "best_validation_bert")

        torch.save(m, os.path.join('./Model/BertCNN/lr_%f_b=%d_LSTM_{%d}.bak'%(args.lr,args.batch_size,epoch)))

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)
    
    #plot train/validation loss versus epoch
    #plot train/validation loss versus epoch
    x = list(range(1, args.epochs+1))
    plt.figure()
    plt.title("train/validation loss versus epoch")
    plt.xlabel("epoch")
    plt.ylabel("Average loss")
    plt.plot(x, train_loss,label="train loss")
    plt.plot(x, test_loss, color='red', label="test loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    #plot train/validation accuracy versus epoch
    x = list(range(1, args.epochs+1))
    plt.figure()
    plt.title("train/validation accuracy versus epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.plot(x, train_acc,label="train accuracy")
    plt.plot(x, test_acc, color='red', label="test accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
