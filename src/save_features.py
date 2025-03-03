import pickle
import torch
import numpy as np
import argparse
import torchvision.models as models
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable, Function
import process_data_weibo as process_data
import copy
from transformers import DistilBertModel, DistilBertTokenizer
def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []
    #length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) -1
        mask_seq = np.zeros(args.sequence_len, dtype = np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)


        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        #length.append(seq_len)
    return word_embedding, mask

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        #self.text = torch.from_numpy(np.array(dataset['post_text']))
        #self.text = torch(dataset['post_text'])
        self.text = dataset['post_text']
        #self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        #self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, labe: %d, Event: %d'
               % (len(self.text), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx], self.event_label[idx]
#def save_features_batch_by_batch(text, mask, all_text_features, all_image_features, save_path='features.pkl'):
 


def save_features_batch_by_batch(args, text,  all_text_features):
    # 用于保存所有批次的特征
    device = "mps"
    #all_features = []  # 用于存储所有批次的特征
    
    # 送入设备
    text = [str(t) for t in text]
    model = BertModel.from_pretrained('bert-base-uncased')  # 使用预训练的 BERT 模型
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text,padding=True, truncation=True, return_tensors='pt')
    inputs = inputs.to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    text_features = outputs.last_hidden_state[:, 0, :]
    #print(model.config.hidden_size)
    #model.bert_fc = nn.Linear(model.config.hidden_size, args.hidden_dim)
    
    # 获取文本特征
    #text_output = model(text, attention_mask=mask)
    
    #text_features = text_output.last_hidden_state[:, 0, :]
    #text_features = text_output.last_hidden_state
    #text_features = model.bert_fc(text_features)

    # 获取图像特征
    #image_features = model.resnet(images)
    #image_features = model.image_fc(image_features)

    
    # 存入列表
    all_text_features.append(text_features.cpu().detach().numpy())
    #all_image_features.append(image_features.cpu().detach().numpy())

    #return all_text_features, all_image_features
    return 

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
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=20, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= True, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    return parser

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    else:
        x = x.to("mps")
    return Variable(x)

def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)
    #print(train['post_text'][:2])
    return train, validate, test
    #print(train[4][0])
    word_vector_path = '../Data/weibo/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)  # W, W2, word_idx_map, vocab
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    args.vocab_size = len(vocab)
    args.sequence_len = max_len
    print("translate data to embedding")

    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, W)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("translate test data to embedding")
    word_embedding, mask = word2vec(test['post_text'], word_idx_map, W)
    test['post_text'] = word_embedding
    test['mask']=mask
    #test[-2]= transform(test[-2])
    word_embedding, mask = word2vec(train['post_text'], word_idx_map, W)
    train['post_text'] = word_embedding
    train['mask'] = mask
    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is "+str(len(train['post_text'])))
    print("Finished loading data ")
    return train, validate, test, W

def main(args):
    print('loading data')
    #    dataset = DiabetesDataset(root=args.training_file)
    #    train_loader = DataLoader(dataset=dataset,
    #                              batch_size=32,
    #                              shuffle=True,
    #                              num_workers=2)

    # MNIST Dataset
    train, validation, test= load_data(args)

    #train, validation = split_train_validation(train,  1)

    #weights = make_weights_for_balanced_classes(train[-1], 15)
    #weights = torch.DoubleTensor(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    

    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test) # not used



    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset = validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    all_text_features = []
    all_train_labels = []
    all_event_labels = []
    #all_image_features = []
    print('building model')

    for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_labels, event_labels = \
                train_data,  \
                to_var(train_labels), to_var(event_labels)
            save_features_batch_by_batch(args, train_text, all_text_features)
            all_train_labels.append(train_labels.cpu().detach().numpy())
            all_event_labels.append(event_labels.cpu().detach().numpy())

    # 把 list 转换为 numpy 数组
    all_text_features = np.concatenate(all_text_features, axis=0)
    all_train_labels = np.concatenate(all_train_labels, axis=0)
    all_event_labels = np.concatenate(all_event_labels, axis=0)
    #all_image_features = np.concatenate(all_image_features, axis=0)
    # 一次性保存
    save_path = 'train_text_features.pkl'
    features_data = {
        'text_features': all_text_features,
        'train_labels': all_train_labels,
        'event_labels': all_event_labels
    }
    
    # 保存打包的数据
    with open(save_path, 'wb') as f:
        pickle.dump(features_data, f)

    print(f"All features saved to {save_path}")

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = '../Data/weibo/train.pickle'
    test = '../Data/weibo/test.pickle'
    output = '../Data/weibo/output/'
    args = parser.parse_args([train, test, output])
    #    print(args)
    main(args)