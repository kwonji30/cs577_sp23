import torch
import torch.nn as nn
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.
class DAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, emb, pos_emb):
        super().__init__()
        self.emb = emb
        self.pos_emb = pos_emb
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)] + 
                                           [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, seq, seq_length, pos):
        inputs_embedded = self.emb(seq)
        pos_embedded = self.pos_emb(pos)
        if inputs_embedded.shape[1] != pos_embedded.shape[1]:
            if pos_embedded.shape[1] > inputs_embedded.shape[1]:
                pos_embedded = pos_embedded[:, :inputs_embedded.shape[1]]        
        inputs_embedded = torch.cat((inputs_embedded, pos_embedded), dim=2)        
        
        out = torch.zeros((seq.shape[0],100))
        for i in range(seq.shape[0]):
            initial = torch.zeros(100)
            for j in range(seq_length[i]):
                initial = torch.add(initial,inputs_embedded[i][j])
                

            mean = initial / seq_length[i]
            out[i] = mean
            
            
        for layer in self.hidden_layers:
            out = layer(out)
            out = F.relu(out)
            out = self.drop(out)
        
        output = self.fc(out)
        return output

class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, emb, pos_emb, bidirectional = True):
        super().__init__()
        self.emb = emb
        self.pos_emb = pos_emb
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        if(self.bidirectional):
            
            self.rnn = nn.RNN(input_dim,
                          hidden_dim,
                          num_layers,
                          bidirectional = True,
                          batch_first=True)
            
            self.drop = nn.Dropout(p=0.5)
            self.fc = nn.Linear(2*hidden_dim, output_dim)            
        else:
            self.rnn = nn.RNN(input_dim,
                          hidden_dim,
                          num_layers,
                          bidirectional = False,
                          batch_first=True)
            self.drop = nn.Dropout(p=0.5)
            self.fc = nn.Linear(hidden_dim, output_dim)   
    
    def forward(self, seq, seq_length, pos):
        inputs_embedded = self.emb(seq)
        pos_embedded = self.pos_emb(pos)
        if inputs_embedded.shape[1] != pos_embedded.shape[1]:
            if pos_embedded.shape[1] > inputs_embedded.shape[1]:
                pos_embedded = pos_embedded[:, :inputs_embedded.shape[1]]
  
        inputs_embedded = torch.cat((inputs_embedded, pos_embedded), dim=2)
    
        inputs_embedded_packed = rnn_utils.pack_padded_sequence(inputs_embedded, seq_length, enforce_sorted = False, batch_first = True)
        outputs_packed, _ = self.rnn(inputs_embedded_packed)
        outputs_unpacked, _ = rnn_utils.pad_packed_sequence(outputs_packed, batch_first=True)

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


class LSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 emb,
                 pos_emb,
                 bidirectional=True):
        super().__init__()
        self.emb = emb
        self.pos_emb = pos_emb
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
        
    
    def forward(self, seq, seq_length, pos):
        inputs_embedded = self.emb(seq)
        pos_embedded = self.pos_emb(pos)
        if inputs_embedded.shape[1] != pos_embedded.shape[1]:
            if pos_embedded.shape[1] > inputs_embedded.shape[1]:
                pos_embedded = pos_embedded[:, :inputs_embedded.shape[1]]
        

        inputs_embedded = torch.cat((inputs_embedded, pos_embedded), dim=2)
        
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

    
 

