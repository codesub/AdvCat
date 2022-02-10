from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.mlp = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
            nn.Dropout(args.dropout_ratio)
        )

    def forward(self, data):
        x = self.mlp(data)
        x = F.log_softmax(x, dim=-1)
        return x


class LSTM(torch.nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.input_size = args.num_features
        self.num_classes = args.num_classes
        self.num_layers = 1
        self.lstm = nn.LSTM(self.input_size, 48, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(48, self.num_classes)
        self.attention = SelfAttention(48)
        self.dropout = nn.Dropout(args.dropout_ratio)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, length=None):
        if length == None:
            length = [50]*x.size(0)
        lengths = torch.LongTensor(length)
        a_lengths, idx = lengths.sort(0, descending=True)
        _, un_idx = torch.sort(idx, dim=0)
        x = x[idx]
        a_packed_input = pack_padded_sequence(input=x, lengths=a_lengths, batch_first=True)
        packed_out, h_n = self.lstm(a_packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=a_lengths[0])
        un_idx = un_idx.cuda()
        output = torch.index_select(out, 0, un_idx)
        x, attn_weights = self.attention(output)
        x = self.fc(x)
        x = self.dropout(x)
        logit = F.log_softmax(x, dim=-1)
        return logit

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights
