from torch.utils.data import Dataset
import pandas as pd
import torch
import nltk
from nltk.tokenize import word_tokenize
import torch.nn.utils.rnn as rnn_utils


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class sentimentDataset(Dataset):
    def __init__(self, data:pd.core.frame.DataFrame, vocab = None, glove = None):
        self.data = data
        self.vocab = None
        if vocab is not None:
            self.vocab = vocab
            
        self.glove = None
        if glove is not None:
            self.glove = glove
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if(self.vocab is None):
            
            label_dict = {'positive': 0, 'negative': 1, 'litigious': 2, 'uncertainty': 3}
            label = label_dict.get(row.Label, -1)

            words = word_tokenize(row.Text.lower())
            
            indices = [self.glove.get_index(w) for w in words if self.glove.has_index_for(w)]
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            return indices_tensor, torch.tensor(label, dtype=torch.long)
        
        else:
            text = row.Text
            label = row.Label

            numericalized_text = self.vocab.numericalize(text)


            label_dict = {'positive': 0, 'negative': 1, 'litigious': 2, 'uncertainty': 3}
            label = label_dict.get(row.Label, -1)
            output = torch.tensor(label, dtype=torch.long)

            return (torch.tensor(numericalized_text, dtype=torch.long), output) 



    
class Vocabulary:
    def __init__(self):
        self.index2str = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>", 4:"<SEP>"}
        self.str2index = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3, "<SEP>":4}
        
    def __len__(self):
        return len(self.index2str)
    
    @staticmethod
    def tokenizer_eng(text):
        return word_tokenize(text.lower())
    
    def build_vocabulary(self, sentence_list):
        index = 5
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in self.str2index:
                    self.index2str[index] = word
                    self.str2index[word] = index
                    index += 1
    
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        result = []
        for token in tokenized_text:
            if token in self.str2index:
                result.append(self.str2index[token])
            else:
                result.append(self.str2index["<UNK>"])
        return result
    
    

    
def collate(batch):
    inputs, labels = zip(*batch)
    new_input = []
    new_lab = []
    for lab,inp in zip(labels,inputs):
        if(len(inp) != 0):
            new_input.append(inp)
            new_lab.append(lab)
    new_lab = torch.LongTensor(new_lab)
    new_input = new_input
    inputs_padded = rnn_utils.pad_sequence(new_input, batch_first=True)



    seq_length = torch.LongTensor([len(seq) for seq in new_input])

    labels_sorted = torch.tensor(new_lab, dtype=torch.float32)

    return inputs_padded, labels_sorted, seq_length