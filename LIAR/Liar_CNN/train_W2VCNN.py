
import argparse
import os
import torch
import torch.optim as optim

import dataset
import model_CNN as model
import training


import numpy as np
from util import *


def main():
    
    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TextCNN')
    #Training args
    parser.add_argument('--data-csv', type=str, default='./IMDB_Dataset.csv',
                        help='file path of training data in CSV format (default: ./train.csv)')
    
    parser.add_argument('--spacy-lang', type=str, default='en', 
                        help='language choice for spacy to tokenize the text')
                        
    parser.add_argument('--pretrained', type=str, default='glove.6B.300d',
                    help='choice of pretrined word embedding from torchtext')              
                        
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training (default: 64)')
    
    parser.add_argument('--val-batch-size', type=int, default=16,
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

    log_f = open('./Logs/W2VCNN/train_lr_%f_b=%d.bak' % (args.lr, args.batch_size), 'w+')
    TITLE = '===== ' + 'train_learning Rate' + str(args.lr) + ' _ Batch Size' + str(args.batch_size) + ' ====='
    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    
    #Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, validset, vocab = dataset.create_tabular_dataset('./train_1.csv',
                                 './test_1.csv',args.spacy_lang, args.pretrained)
    
    #%%Show some example to show the dataset
    print("Show some examples from train/valid..",file=log_f, flush=True)
    print(trainset[0].text,  trainset[0].label)
    print(validset[0].text,  validset[0].label)


    train_iter, valid_iter = dataset.create_data_iterator(args.batch_size, args.val_batch_size,
                                                         trainset, validset,device)
                
    #%%Create
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = model.textCNN(vocab, args.out_channel, kernels, args.dropout , args.num_class).cuda()
    # print the model summery
    print(m)
    best_test_acc = -1
    
    #optimizer
    optimizer = optim.Adam(m.parameters(), lr=args.lr)
    
    for epoch in range(0, args.epochs+1):
        print('================== This is the %d epoch===================== ' % (epoch), file=log_f, flush=True)
        print('======This is the TRAINing part====== ', file=log_f, flush=True)
        #train loss
        trainy = []
        trainyhat_probs = []
        trainyhat_classes = []
        tr_loss, tr_acc,trainy,trainyhat_probs,trainyhat_class = training.train(m, device, train_iter, optimizer, epoch, args.epochs,trainy,trainyhat_probs,trainyhat_classes)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc), file=log_f, flush=True)

        trainy = np.array(trainy)
        trainyhat_probs = np.array(trainyhat_probs)
        trainyhat_classes = np.array(trainyhat_classes)

        performance(trainy, trainyhat_classes,trainyhat_probs , log_f)

        print('======This is the TESTing part====== ', file=log_f, flush=True)
        testy = []
        yhat_probs = []
        yhat_classes = []
        ts_loss, ts_acc,testy,yhat_probs,yhat_classes = training.valid(m, device, valid_iter,testy,yhat_probs,yhat_classes)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, ts_loss, ts_acc), file=log_f, flush=True)

        performance(testy, yhat_classes, yhat_probs, log_f)

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_test_acc))
            torch.save(m.state_dict(), "best_validation")

        torch.save(m, os.path.join('./Model/W2VCNN/lr_%f_b=%d_LSTM_{%d}.bak'%(args.lr,args.batch_size,epoch)))


if __name__ == '__main__':
    main()

