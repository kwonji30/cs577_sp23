import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.FloatTensor)
#torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cuda")
import warnings
warnings.filterwarnings("ignore")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api

from neural_archs import LSTM
from utils import sentimentDataset
from utils import Vocabulary
from samsharp import SAMSharp as SAM
from utils import collate
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
nltk.download('punkt')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f')

    # TODO: Read off the WiC dataset files from the `WiC_dataset' directory
    # (will be located in /homes/cs577/WiC_dataset/(train, dev, test))
    # and initialize PyTorch dataloader appropriately
    # Take a look at this page
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # and implement a PyTorch Dataset class for the WiC dataset in
    # utils.py

    train_df = pd.read_csv('train.csv', sep=',', header=0, names=['Text', 'Label'])
    train_df = train_df.sample(frac=0.05)
    valid_df = pd.read_csv('val.csv', sep=',', header=0, names=['Text', 'Label'])
    valid_df = valid_df.sample(frac=0.05)
    test_df = pd.read_csv('test.csv', sep=',', header=0, names=['Text', 'Label'])
    test_df = test_df.sample(frac=0.05)
    

    glove = api.load("glove-wiki-gigaword-50")
    sentence_list = []
    for ind, row in train_df.iterrows():
        sentence_list.append(row.Text.lower())

    vocab = Vocabulary()
    vocab.build_vocabulary(sentence_list)        
    emb = nn.Embedding.from_pretrained(torch.FloatTensor(glove.vectors))
    emb = emb.to(torch_device)

    train_dataset = sentimentDataset(train_df, vocab = None, glove = glove)
    indices = []

    for ind,row in test_df.iterrows():
        words = word_tokenize(row.Text.lower())

        indices.append(torch.tensor([glove.get_index(w) for w in words if glove.has_index_for(w)], dtype=torch.long))
    # pad the inputs with zeros to make them the same length
    test_padded = rnn_utils.pad_sequence(indices, batch_first=True)
    # get the sequence lenghts of the inputs
    test_seq_length = torch.LongTensor([len(seq) for seq in indices])

    valid_indices = []
    test_indices = []
    label_dict = {'positive': 0, 'negative': 1, 'litigious': 2, 'uncertainty': 3}

    valid_labels = valid_df['Label'].apply(lambda x: label_dict.get(x, -1))
    test_labels = test_df['Label'].apply(lambda x: label_dict.get(x, -1))

    for ind,row in valid_df.iterrows():
        words = word_tokenize(row.Text.lower())
        valid_indices.append(torch.tensor([glove.get_index(w) for w in words if glove.has_index_for(w)], dtype=torch.long))
    for ind,row in test_df.iterrows():
        words = word_tokenize(row.Text.lower())
        test_indices.append(torch.tensor([glove.get_index(w) for w in words if glove.has_index_for(w)], dtype=torch.long))
        
    new_input = []
    new_lab = []
    for lab,inp in zip(valid_labels,valid_indices):
        if(len(inp) != 0):
            new_input.append(inp)
            new_lab.append(lab)
    new_lab = torch.LongTensor(new_lab)
    valid_inputs_padded = rnn_utils.pad_sequence(new_input, batch_first=True)

    valid_seq_length = torch.LongTensor([len(seq) for seq in new_input])

    valid_padded = rnn_utils.pad_sequence(valid_inputs_padded, batch_first=True)
    valid_labels_sorted = torch.tensor(new_lab, dtype=torch.float32)
    
    new_input = []
    new_lab = []
    for lab,inp in zip(test_labels,test_indices):
        if(len(inp) != 0):
            new_input.append(inp)
            new_lab.append(lab)
    new_lab = torch.LongTensor(new_lab)
    test_inputs_padded = rnn_utils.pad_sequence(new_input, batch_first=True)

    test_seq_length = torch.LongTensor([len(seq) for seq in new_input])
    # pad the inputs with zeros to make them the same length
    test_padded = rnn_utils.pad_sequence(test_inputs_padded, batch_first=True)
    test_labels_sorted = torch.tensor(new_lab, dtype=torch.float32)

    lr = 0.001
    epochs = 50

    model = LSTM(50, 128, 1, 2, emb, bidirectional=True).to(torch_device)
    



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, drop_last = True, collate_fn = collate)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = SAM(params=model.parameters(),lr=lr, rho=0.05)
    criterion = nn.CrossEntropyLoss().to(torch_device)
    lamda = 1.5

    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / y_true.numel()) * 100
        return acc
    
    train_losses = []
    test_losses = []
    # TODO: Training and validation loop here
    val_max = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n------")
        ### Training
        train_loss, train_acc = 0, 0
        for inputs_packed, labels, seq_lengths in train_loader:
            model.train()
            inputs_packed = inputs_packed.to(torch_device)
            labels = labels.to(torch_device)
            
            def forward():
                optimizer.zero_grad()
                output = model(inputs_packed, seq_lengths)
                return output.squeeze()
            

            def closure(outputs=None):

                optimizer.zero_grad()
                output = model(inputs_packed, seq_lengths).squeeze()
                if outputs is None:
                    loss = criterion(output, labels)
                else:
                    loss =  lamda * torch.abs(criterion(output, labels) - criterion(outputs, labels)) + criterion(output, labels)
                
                loss.backward()
                
                return loss, output
#             def closure():
#                 optimizer.zero_grad()
#                 output = model(inputs_packed, seq_lengths)
#                 loss = criterion(output.squeeze(), labels)
#                 loss.backward()
#                 return loss, output
            loss, outputs = optimizer.step(closure, forward)
#             outputs = outputs.squeeze()
            
            #logits = model(inputs_packed, seq_lengths).squeeze()
            pred = torch.round(torch.sigmoid(outputs))

            #loss = criterion(logits, labels)
            train_loss += loss
            train_acc += accuracy_fn(pred, labels)

            #optimizer.zero_grad()

            #loss.backward()

            #optimizer.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        ### Testing
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.no_grad():
            # Forward pass
            test_logits = model(valid_inputs_padded.to(torch_device), valid_seq_length).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # Calculate loss and accuracy
            valid_labels = valid_labels_sorted.to(torch_device)
            test_loss = nn.CrossEntropyLoss()
            test_loss = test_loss(test_logits, valid_labels)
            test_acc = accuracy_fn(test_pred, valid_labels)
            if test_acc > val_max:
                print("Saving Model...\n")
                torch.save(model.state_dict(), 'best.pt')
                val_max = test_acc
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        print(f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}% | Test loss: {test_loss/len(valid_labels):.4f}, Test Acc: {test_acc:.4f}%\n")

    
    model.load_state_dict(torch.load('best.pt'))


    model.eval()
    logits = model(test_inputs_padded.to(torch_device), test_seq_length).squeeze()
    pred = torch.round(torch.sigmoid(logits)).to('cpu')

    # compare the labels and calculate accuracy
    total = test_labels_sorted.numel()
    correct = torch.eq(test_labels_sorted, pred).sum().item()
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")     