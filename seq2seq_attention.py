import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # SHAPE(x): (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # SHAPE(embedding): (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # SHAPE(encoder_states): (seq_length, batch, num_directions * hidden_size)
        # SHAPE(hidden): (num_layers * num_directions, batch, hidden_size)
        # SHAPE(cell): (num_layers * num_directions, batch, hidden_size)

        # Forward and backward hidden state of the last time state is concatenated
        # in (batch_size, hidden_size*2)
        hidden = self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        cell = self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))
        # SHAPE(hidden/cell): (batch_size, hidden_size)
        # Gotta unsqueeze ig so...............
        return encoder_states, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_size*3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_states, hidden, cell):
        # SHAPE(encoder_states): (seq_length, batch, num_directions * hidden_size)
        # SHAPE(hidden): (batch, hidden_size)
        # SHAPE(cell): (batch, hidden_size)

        seq_length = encoder_states.shape[0]
        batch_size = encoder_states.shape[1]
        
        hidden = hidden.unsqueeze(1)
        # SHAPE(hidden): (batch, 1, hidden_size)
        
        hidden = hidden.repeat(1, seq_length, 1)
        # SHAPE(hidded): (batch, seq_length, hidden_size)

        encoder_states = encoder_states.permute(1, 0, 2)
        # SHAPE(encoder_states): (batch, seq_length, num_directions * hidden_size)
        # print(encoder_states.shape, hidden.shape)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_states), dim=2)))
        # SHAPE(energy): (batch, seq_length, hidden_size)

        attention = self.v(energy).squeeze(2)
        # shape(attention): (batch, seq_length)

        attention = F.softmax(attention, dim=1)
        # shape(attention): (batch, seq_length)

        return attention


class Decoder(nn.Module):
    def __init__(self, attention, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.attention = attention

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, 1)

        self.fc = nn.Linear(hidden_size*3 + embedding_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x, encoder_states, hidden, cell):
        # SHAPE(encoder_states): (seq_length, batch, num_directions * hidden_size)
        # SHAPE(hidden): (batch, hidden_size)
        # SHAPE(cell): (batch, hidden_size)

        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        attention = self.attention(encoder_states, hidden, cell)
        # attention: (batch, seq_length)

        attention = attention.unsqueeze(1)
        # attention: (batch, 1, seq_length)

        encoder_states = encoder_states.permute(1, 0, 2)
        # SHAPE(encoder_states): (batch, seq_length, num_directions * hidden_size)

        weighted = torch.bmm(attention, encoder_states)
        # weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((weighted, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # outputs = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]

        embedding = embedding.squeeze(0)
        outputs = outputs.squeeze(0)
        weighted = weighted.squeeze(0)

        predictions = self.fc(torch.cat((outputs, weighted, embedding), dim = 1))
        # predictions: (N, hidden_size)

        return predictions, hidden.squeeze(0), cell.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, vocab_size, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
