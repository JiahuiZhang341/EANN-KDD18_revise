import numpy as np
import argparse
import time, os
# import random
import sys
sys.path.append('/Users/jiahuizhang/Documents/GitHub/EANN-KDD18_revise/src')  # 确保模块路径被加入

import process_data_weibo as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from transformers import BertModel, BertTokenizer

import torchvision.datasets as dsets
import torchvision.transforms as transforms

#from logger import Logger

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
import pickle
writer = SummaryWriter('./log')

class TextFeaturesDataset(Dataset):
    def __init__(self, file_path):
        # 从文件中加载特征数据
        with open(file_path, 'rb') as f:
            features_data = pickle.load(f)
        
        # 提取数据
        self.text_features = features_data['text_features']
        self.train_labels = features_data['train_labels']
        self.event_labels = features_data['event_labels']

    def __len__(self):
        # 返回样本数量
        return len(self.text_features)

    def __getitem__(self, idx):
        # 返回索引对应的特征，通常可以选择文本特征和对应标签
        # 假设文本特征是一个张量，返回的是该样本的特征
        # 返回一个样本的特征和标签
        text_feature = torch.tensor(self.text_features[idx], dtype=torch.float32)
        train_label = torch.tensor(self.train_labels[idx], dtype=torch.long)  # 假设是分类任务
        event_label = torch.tensor(self.event_labels[idx], dtype=torch.long)  # 假设是分类任务
        return text_feature, train_label, event_label



def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    else:
        x = x.to("mps")
    return Variable(x)

class ReverseLayerF(Function):

    #@staticmethod
    def forward(self, x, lambd):
        self.lambd = lambd
        return x.view_as(x)

    #@staticmethod
    def backward(self, grad_output):
        return grad_output.neg() * self.lambd, None

def grad_reverse(x, lambd =1.0):
    return ReverseLayerF.apply(x, lambd)


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # TEXT Bert
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # 使用预训练的 BERT 模型
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, self.args.hidden_dim)

        self.dropout = nn.Dropout(args.dropout)

        #IMAGE
        #hidden_size = args.hidden_dim
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉最后的分类层
        self.image_fc = nn.Linear(2048, self.args.hidden_dim) 


        ###social context
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1',  nn.Linear(self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        #self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        #self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        #self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        #self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        ####Image and Text Classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(self.hidden_size, 2))
        self.modal_classifier.add_module('m_softmax', nn.Softmax(dim=1))


    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        #x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text):
        text = self.bert_fc(text)
        text = self.dropout(text)
        ### Class
        #class_output = self.class_classifier(text_image)
        class_output = self.class_classifier(text)
        ## Domain
        reverse_feature = grad_reverse(text, lambd=args.lambd)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

def main(args):
    print('loading data')
    #读取已经提取好的数据特征(这里只有text)
    ## 创建 Dataset 对象
    train_dataset = TextFeaturesDataset(file_path='train_text_features.pkl')
    valid_dataset = TextFeaturesDataset(file_path='valid_text_features.pkl')
    test_dataset = TextFeaturesDataset(file_path='test_text_features.pkl')
    ## 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('building model')
    model = CNN_Fusion(args)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()
    else:
        model.to("mps")


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    #Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
    #                             lr= args.learning_rate)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
                                  #lr=args.learning_rate)
    #scheduler = StepLR(optimizer, step_size= 10, gamma= 1)

    iter_per_epoch = len(train_loader)
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_loss = 100
    best_validate_dir = ''

    print('training model')
    adversarial = True
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        #lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 3e-5

        optimizer.lr = lr
        #rgs.lambd = lambd

        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []
        ## 遍历 DataLoader 获取批次数据
        for i, (batch_data, train_labels, event_labels) in enumerate(train_loader):
            batch_data, train_labels, event_labels = to_var(batch_data), to_var(train_labels), to_var(event_labels)
             # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model(batch_data)
            # ones = torch.ones(text_output.size(0))
            # ones_label = to_var(ones.type(torch.LongTensor))
            # zeros = torch.zeros(image_output.size(0))
            # zeros_label = to_var(zeros.type(torch.LongTensor))

            #modal_loss = criterion(text_output, ones_label)+ criterion(image_output, zeros_label)

            class_loss = criterion(class_outputs, train_labels)
            domain_loss = criterion(domain_outputs, event_labels)
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            #domain_cost_vector.append(domain_loss.data[0])
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        
            # if i == 0:
            #     train_score = to_np(class_outputs.squeeze())
            #     train_pred = to_np(argmax.squeeze())
            #     train_true = to_np(train_labels.squeeze())
            # else:
            #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
            #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
            #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)


        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(valid_loader):
            validate_text, validate_labels, event_labels = to_var(validate_data), to_var(validate_labels), to_var(event_labels)
            validate_outputs, domain_outputs = model(validate_text)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels)
            #domain_loss = criterion(domain_outputs, event_labels)
                #_, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append( vali_loss.item())
                #validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        print ('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, validate loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
                % (
                epoch + 1, args.num_epochs,  np.mean(cost_vector), np.mean(class_cost_vector), np.mean(vali_cost_vector),
                    np.mean(acc_vector),   validate_acc, ))
        # 计算每个 epoch 的平均损失
        avg_loss = vali_loss / len(valid_loader)
    
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('accuracy', validate_accuracy, epoch)

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)
            best_validate_dir = args.output_file + str(epoch + 1) + '_text.pkl'
            torch.save(model.state_dict(), best_validate_dir)
    writer.close()

    duration = time.time() - start_time

    # Test the Model
    print('testing model')
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to("mps")
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_labels = to_var(test_data), to_var(test_labels)
        test_outputs, _= model(test_text)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')

    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
   
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred, digits=3)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

    print('Saving results')


def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    #parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.3, help='')
    parser.add_argument('--filter_num', type=int, default=20, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= True, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    return parser

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = '../Data/weibo/train.pickle'
    test = '../Data/weibo/test.pickle'
    output = '../Data/weibo/output/'
    args = parser.parse_args([train, test, output])
    #    print(args)
    main(args)
