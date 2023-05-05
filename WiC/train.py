import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cuda")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api

from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset
from utils import Vocabulary
from utils import POSvocab
from utils import collate
import nltk
from nltk import pos_tag
from samadam import SAMAdam as SAM
import pandas as pd
from nltk.tokenize import word_tokenize
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    
    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='dan', type=str)
    parser.add_argument('--rnn_bidirect', default=True, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='glove', type=str)

    args = parser.parse_args()

    # TODO: Read off the WiC dataset files from the `WiC_dataset' directory
    # (will be located in /homes/cs577/WiC_dataset/(train, dev, test))
    # and initialize PyTorch dataloader appropriately
    # Take a look at this page
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # and implement a PyTorch Dataset class for the WiC dataset in
    # utils.py

    train_df = pd.concat([pd.read_csv('~/WiC_dataset/train/train.data.txt', sep='\t', header=None, names=['Target', 'PoS', 'Index', 'Context_1', 'Context_2']),
                      pd.read_csv('~/WiC_dataset/train/train.gold.txt', sep='\t', header=None, names=['Label'])], axis=1)

    valid_df = pd.concat([pd.read_csv('~/WiC_dataset/dev/dev.data.txt', sep='\t', header=None, names=['Target', 'PoS', 'Index', 'Context_1', 'Context_2']),
                      pd.read_csv('~/WiC_dataset/dev/dev.gold.txt', sep='\t', header=None, names=['Label'])], axis=1)
    
    test_df = pd.read_csv('~/WiC_dataset/test/test.data.txt', sep='\t', header=None, names=['Target', 'PoS', 'Index', 'Context_1', 'Context_2'])
                      
    
    valid_df['Combined'] = valid_df['Context_1'] + ' ' + valid_df['Context_2'] + ' ' + valid_df['Target']
    train_df['Combined'] = train_df['Context_1'] + ' ' + train_df['Context_2'] + ' ' + train_df['Target']
    test_df['Combined'] = test_df['Context_1'] + ' ' + test_df['Context_2'] + ' ' + test_df['Target'] 
    train_df["PosTags"] = train_df.apply(lambda row: [tag for _, tag in pos_tag(word_tokenize(row["Combined"].lower()))], axis=1)
    valid_df["PosTags"] = valid_df.apply(lambda row: [tag for _, tag in pos_tag(word_tokenize(row["Combined"].lower()))], axis=1)
    test_df["PosTags"] = test_df.apply(lambda row: [tag for _, tag in pos_tag(word_tokenize(row["Combined"].lower()))], axis=1)
    
    
    if args.init_word_embs == "glove":
        # TODO: Feed the GloVe embeddings to NN modules appropriately
        # for initializing the embeddings
        glove = api.load("glove-wiki-gigaword-50")
        sentence_list = []
        for index, row in train_df.iterrows():
            sentence_list.append(row.PosTags)

        pos_vocab = POSvocab()
        pos_vocab.build_vocabulary(sentence_list)        
        
        emb = nn.Embedding.from_pretrained(torch.FloatTensor(glove.vectors))
        pos_emb = nn.Embedding(num_embeddings = len(pos_vocab), embedding_dim = 50)
        
        train_dataset = WiCDataset(train_df, vocab = None, glove = glove, pos_vocab=pos_vocab)
        
        indices = []
        numericalized_pos = []
        for ind,row in test_df.iterrows():
            words = word_tokenize(row.Combined.lower())
            pos_tags = pos_vocab.numericalize(row.PosTags)
            if len(pos_tags) > len(words):
                pos_tags = pos_tags[:len(words)]
            numericalized_pos.append(torch.tensor(pos_tags,dtype=torch.long))
            indices.append(torch.tensor([glove.get_index(w) for w in words if glove.has_index_for(w)], dtype=torch.long))
        # pad the inputs with zeros to make them the same length
        test_pos_padded = rnn_utils.pad_sequence(numericalized_pos, batch_first=True)
        test_padded = rnn_utils.pad_sequence(indices, batch_first=True)
        # get the sequence lenghts of the inputs
        test_seq_length = torch.LongTensor([len(seq) for seq in indices])
        
        indices = []
        numericalized_pos = []
        labels = valid_df['Label'].apply(lambda x: 1 if x == 'T' else 0)
        valid_labels = torch.tensor(labels, dtype=torch.float32)
        for ind,row in valid_df.iterrows():
            words = word_tokenize(row.Combined.lower())
            pos_tags = pos_vocab.numericalize(row.PosTags)
            if len(pos_tags) > len(words):
                pos_tags = pos_tags[:len(words)]
            numericalized_pos.append(torch.tensor(pos_tags,dtype=torch.long))
            indices.append(torch.tensor([glove.get_index(w) for w in words if glove.has_index_for(w)], dtype=torch.long))
        # pad the inputs with zeros to make them the same length
        valid_pos_padded = rnn_utils.pad_sequence(numericalized_pos, batch_first=True)
        valid_padded = rnn_utils.pad_sequence(indices, batch_first=True)
        # get the sequence lenghts of the inputs
        valid_seq_length = torch.LongTensor([len(seq) for seq in indices])

    else:
        sentence_list = []
        for index, row in train_df.iterrows():
            sentence_list.append(row.Combined)
            
        vocab = Vocabulary()
        vocab.build_vocabulary(sentence_list)

        for index, row in train_df.iterrows():
            sentence_list.append(row.PosTags)

        pos_vocab = POSvocab()
        pos_vocab.build_vocabulary(sentence_list)    
        
        train_dataset = WiCDataset(train_df, vocab=vocab, pos_vocab=pos_vocab)
    
        emb = nn.Embedding(num_embeddings = len(vocab), embedding_dim = 50)
        pos_emb = nn.Embedding(num_embeddings = len(pos_vocab), embedding_dim = 50)
        
        indices = []
        numericalized_pos = []
        for ind,row in test_df.iterrows():
            context1 = row.Context_1
            context2 = row.Context_2
            target_word = row.Target
            pos_tags = pos_vocab.numericalize(row.PosTags)
            numericalized_pos.append(torch.tensor(pos_tags, dtype= torch.long))

            numericalized_context = vocab.numericalize(context1)
            numericalized_context += vocab.numericalize(context2)
            indices.append(torch.LongTensor(numericalized_context))

        # pad the inputs with zeros to make them the same length
        test_pos_padded = rnn_utils.pad_sequence(numericalized_pos, batch_first=True)
        test_padded = rnn_utils.pad_sequence(indices, batch_first=True)
        # get the sequence lenghts of the inputs
        test_seq_length = torch.LongTensor([len(seq) for seq in indices])


        indices = []
        numericalized_pos = []
        labels = valid_df['Label'].apply(lambda x: 1 if x == 'T' else 0)
        valid_labels = torch.tensor(labels, dtype=torch.float32)
        for ind,row in valid_df.iterrows():
            context1 = row.Context_1
            context2 = row.Context_2
            target_word = row.Target
            pos_tags = pos_vocab.numericalize(row.PosTags)
            numericalized_pos.append(torch.tensor(pos_tags, dtype= torch.long))

            numericalized_context = vocab.numericalize(context1)
            numericalized_context += vocab.numericalize(context2)
            indices.append(torch.LongTensor(numericalized_context))


        valid_pos_padded = rnn_utils.pad_sequence(numericalized_pos, batch_first=True)
        valid_padded = rnn_utils.pad_sequence(indices, batch_first=True)
        valid_seq_length = torch.LongTensor([len(seq) for seq in indices])

        
        
    lr = 0.001
    epochs = 60
    
    if args.neural_arch == "dan":
        epochs = 150
        model = DAN(100, 1024, 1, 2, emb, pos_emb).to(torch_device)
    elif args.neural_arch == "rnn":
        if args.rnn_bidirect:
            model = RNN(100, 128, 1, 2, emb, pos_emb, bidirectional=True).to(torch_device)
        else:
            model = RNN(100, 128, 1, 2, emb, pos_emb, bidirectional=False).to(torch_device)
    elif args.neural_arch == "lstm":
        if args.rnn_bidirect:
            model = LSTM(100, 128, 1, 2, emb, pos_emb, bidirectional=True).to(torch_device)
        else:
            model = LSTM(100, 128, 1, 2, emb, pos_emb, bidirectional=False).to(torch_device)
    



    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, drop_last = True, collate_fn = collate)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = SAM(params=model.parameters(),lr=lr, rho=0.1).to(torch_device)
    criterion = nn.BCEWithLogitsLoss().to(torch_device)
    
    # Calculate accuracy (a classification metric)
    def accuracy_fn(y_pred, y_true):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / y_true.numel()) * 100
        return acc

    #import matplotlib.pyplot as plt

    train_losses = []
    test_losses = []
    # TODO: Training and validation loop here
    val_max = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n------")
        ### Training
        train_loss, train_acc = 0, 0
        for inputs_packed, labels, seq_lengths, pos in train_loader:
            inputs_packed.to(torch_device)
            labels.to(torch_device)
            seq_lengths.to(torch_device)
            pos.to(torch_device)
            
            model.train()
            # Forward Pass
            def closure():
                optimizer.zero_grad()
                output = model(inputs_packed, seq_lengths, pos)
                loss = criterion(output.squeeze(), labels)
                loss.backward()
                return loss, output
            loss, outputs = optimizer.step(closure)
            outputs = outputs.squeeze()
            #logits = model(inputs_packed, seq_lengths, pos).squeeze()
            pred = torch.round(torch.sigmoid(outputs))
            # Loss 
            #loss = criterion(logits, labels)
            train_loss += loss
            train_acc += accuracy_fn(pred, labels)
            # Zero the graident
            #optimizer.zero_grad()
            # Perform backpropagation
            #loss.backward()
            # Perform gradient descent
            #optimizer.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        ### Testing
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.no_grad():
            # Forward pass
            test_logits = model(valid_padded, valid_seq_length, valid_pos_padded).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # Calculate loss and accuracy
            test_loss = F.binary_cross_entropy(torch.sigmoid(test_logits), valid_labels)
            test_acc = accuracy_fn(test_pred, valid_labels)
            if train_acc > 90 and test_acc > val_max:
                print("Saving Model...\n")
                torch.save(model.state_dict(), 'best.pt')
                val_max = test_acc
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        print(f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}% | Test loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%\n")

    #plt.plot(train_losses[1:], label='Training Loss')
    #plt.plot(test_losses[1:], label='Validation Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()
    #plt.savefig('learning_curve.png')
    
    # TODO: Testing loop
    # Write predictions (F or T) for each test example into test.pred.txt
    # One line per each example, in the same order as test.data.txt.
    
    model.load_state_dict(torch.load('best.pt'))


    model.eval()
    logits = model(test_padded, test_seq_length, test_pos_padded).squeeze()
    pred = torch.round(torch.sigmoid(logits))   

    with open("test.pred.txt", "w") as outfile:
        for prediction in pred:
            if prediction.item() == 1.0:
                outfile.write('T' + "\n")
            else:
                outfile.write('F' + "\n")
                
        # read the true labels from test.gold.txt
    with open("test.gold.txt", "r") as goldfile:
        gold_labels = [line.strip() for line in goldfile]

    # read the predicted labels from test.pred.txt
    with open("test.pred.txt", "r") as predfile:
        pred_labels = [line.strip() for line in predfile]

    # compare the labels and calculate accuracy
    total = len(gold_labels)
    correct = sum(1 for i in range(total) if gold_labels[i] == pred_labels[i])
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")            
    