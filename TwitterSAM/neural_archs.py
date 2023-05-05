import torch
import torch.nn as nn
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F



class LSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 emb,
                 bidirectional=True):
        super().__init__()
        self.emb = emb
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if(self.bidirectional):
            
            self.rnn = nn.LSTM(input_dim,
                          hidden_dim,
                          num_layers,
                          bidirectional = True,
                          batch_first=True)
            
            self.drop = nn.Dropout(p=0.5)
            self.fc = nn.Linear(2*hidden_dim, output_dim)            
        else:
            self.rnn = nn.LSTM(input_dim,
                          hidden_dim,
                          num_layers,
                          bidirectional = False,
                          batch_first=True)
            
            self.drop = nn.Dropout(p=0.5)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, seq, seq_length):
        inputs_embedded = self.emb(seq)
        
        inputs_embedded_packed = rnn_utils.pack_padded_sequence(inputs_embedded, seq_length, enforce_sorted = False,
                                                  batch_first = True)

        
        outputs_packed, _ = self.rnn(inputs_embedded_packed)
        outputs_unpacked, _ = rnn_utils.pad_packed_sequence(outputs_packed,
                                                            batch_first=True)
        
        if(self.bidirectional):
            out_forward = outputs_unpacked[range(len(outputs_unpacked)), seq_length - 1, :self.hidden_dim]
            out_reverse = outputs_unpacked[:, 0, self.hidden_dim:]
            final_out = torch.cat((out_forward, out_reverse), 1)
        else:
            idx = (seq_length - 1).view(-1, 1, 1).expand(len(seq_length), 1, self.rnn.hidden_size)
            final_out = outputs_unpacked.gather(1, idx).squeeze(1)
        
        output = self.drop(final_out)
        output = self.fc(output)
        return output

    
 

