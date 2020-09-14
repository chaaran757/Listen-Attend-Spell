import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.autograd import Variable

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        mask = torch.from_numpy(np.random.binomial(1, dropout, (x.shape[1], x.shape[2])) / dropout).to(device)
        mask = Variable(mask, requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EmbeddingDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embed, words, p):
        mask = torch.from_numpy(np.random.binomial(1, p, size=(embed.weight.data.shape[0])) / p).to(device)
        mask = Variable(mask, requires_grad=False)
        masked_embedding_weights = mask.unsqueeze(1) * embed.weight.data
        masked_embedding_weights = masked_embedding_weights.to(device)
        embedding = torch.nn.functional.embedding(words.long(), masked_embedding_weights, padding_idx=0).float()
        return embedding


class pBlstm(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBlstm, self).__init__()
        self.Blstm = nn.LSTM(input_dim * 2, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, input, input_lens):
        batch_size = input.shape[0]
        feature_dim = input.shape[2]

        input_mod = []

        for i in range(input.shape[0]):
            if (input_lens[i] % 2) != 0:
                input_mod.append(input[i, :input_lens[i] - 1, :])
                input_lens[i] -= 1
            else:
                input_mod.append(input[i, :input_lens[i]])

        input_mod = [input_mod[i].view((int(input_lens[i] / 2), feature_dim * 2)) for i in range(len(input_mod))]

        input_lens = torch.tensor([input_mod[i].shape[0] for i in range(len(input_mod))])

        input_mod = torch.nn.utils.rnn.pad_sequence(input_mod, batch_first=True)
        input_mod = torch.nn.utils.rnn.pack_padded_sequence(input_mod, input_lens, batch_first=True,
                                                            enforce_sorted=False)

        output = self.Blstm(input_mod)[0]

        output, out_lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, out_lens


class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Listener, self).__init__()
        self.Blstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        self.pBlstm1 = pBlstm(hidden_dim * 2, hidden_dim)
        self.pBlstm2 = pBlstm(hidden_dim * 2, hidden_dim)
        self.pBlstm3 = pBlstm(hidden_dim * 2, hidden_dim)

        self.variational_dropout = VariationalDropout()

        self.key = nn.Linear(hidden_dim * 2, key_size)
        self.value = nn.Linear(hidden_dim * 2, value_size)

    def forward(self, input, input_lens, is_train=True):
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(input, input_lens, batch_first=True, enforce_sorted=False)
        output = self.Blstm(packed_data)[0]
        output, out_lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output, out_lens = self.pBlstm1(output, out_lens)
        output, out_lens = self.pBlstm2(output, out_lens)
        output, out_lens = self.pBlstm3(output, out_lens)

        if is_train:
            output = self.variational_dropout(output, 0.50)

        key = self.key(output.float())
        value = self.value(output.float())

        return key, value, out_lens


class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear(512,128)

    def forward(self, query, keys, values, out_lens):
        # compute energy
        scale = 1.0 / np.sqrt(128)
        # query = self.linear(query)
        query = query.unsqueeze(1)  # [B,Q] -> [B,1,Q]
        energy = torch.bmm(query, keys.transpose(1, 2))
        attn = energy.mul_(scale).squeeze(1)
        mask = torch.zeros([query.shape[0], keys.shape[1]], dtype=torch.bool).to(device)
        for i in range(out_lens.shape[0]):
            mask[i, :out_lens[i]] = True
        attn = attn.masked_fill_(~mask, float('-inf'))
        attn = self.softmax(attn)
        # weight values
        context = torch.sum(values * attn.unsqueeze(2).repeat(1, 1, values.size(2)), dim=1)

        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_dim, num_layers=1):
        super(Decoder, self).__init__()
        # Hyper parameters
        # embedding + output
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        # rnn
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Components
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_dropout = EmbeddingDropout()
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim +
                                 self.context_dim, self.hidden_size)]
        for l in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.hidden_size, self.context_dim)]
        self.attention = Attention()
        self.mlp = nn.Sequential(
            nn.Linear(self.context_dim * 2,
                      self.context_dim),
            nn.Tanh(),
            nn.Linear(self.context_dim, self.vocab_size))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.hidden_size if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

    def forward(self, padded_input, text_lens, out_lens, criterion, key, value, tf_prob, is_train=True):
        ys = [y[y != PAD_token] for y in padded_input]
        eos = ys[0].new([EOS_token])
        sos = ys[0].new([SOS_token])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        ys_in_pad = pad_list(ys_in, EOS_token)
        ys_out_pad = pad_list(ys_out, PAD_token)
        assert ys_in_pad.size() == ys_out_pad.size()
        batch_size = ys_in_pad.size(0)
        output_length = ys_in_pad.size(1)

        h_list = [self.zero_state(padded_input)]
        c_list = [self.zero_state(padded_input)]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(padded_input, H=self.context_dim))
            c_list.append(self.zero_state(padded_input, H=self.context_dim))
        att_c = self.zero_state(padded_input,
                                H=self.context_dim)
        y_all = []
        attentions = []

        embedded = self.embedding_dropout(self.embedding, ys_in_pad.long(), 0.50).to(device)
        for t in range(output_length):
            # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
            rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l - 1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]
            # step 2. attention: c_i = AttentionContext(s_i,h)
            att_c = self.attention(rnn_output, key, value, out_lens)
            # step 3. concate s_i and c_i, and input to MLP
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)

            if is_train:
                tf = np.random.binomial(1, tf_prob)
                if tf and t < (output_length - 1):
                    embedded[:, t + 1, :] = self.embedding(torch.max(predicted_y_t, dim=1)[1])

            y_all.append(predicted_y_t)

        y_all = torch.stack(y_all, dim=1)  # N x To x C

        y_all = y_all.transpose(1, 2)
        mask = torch.zeros([len(text_lens), text_lens.max() + 1], dtype=torch.bool)
        for i in range(len(text_lens)):
            text_lens[i] += 2
            mask[i, :text_lens[i] - 1] = True
        mask = mask.to(device)
        loss = criterion(y_all, ys_out_pad.long())
        masked_loss = loss.masked_fill_(~mask, 0).sum()

        return masked_loss

    def forward_test(self, text, key, value, out_lens):
        char_indices = []
        max_length = 250

        h_list = [self.zero_state(text.float())]
        c_list = [self.zero_state(text.float())]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(text.float(), H=self.context_dim))
            c_list.append(self.zero_state(text.float(), H=self.context_dim))
        att_c = self.zero_state(text.float(),
                                H=self.context_dim)

        embedded = self.embedding(text.long())
        for t in range(max_length):
            # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
            rnn_input = torch.cat((embedded[:, 0, :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l - 1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]
            # step 2. attention: c_i = AttentionContext(s_i,h)
            att_c = self.attention(rnn_output, key, value, out_lens)
            # step 3. concate s_i and c_i, and input to MLP
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)
            predicted_y_t = F.log_softmax(predicted_y_t, dim=1)
            char_ind = torch.max(predicted_y_t, dim=1)[1]
            char_indices.append(char_ind)

            embedded = self.embedding(char_ind.view(char_ind.shape[0], 1)).to(device)

        return char_indices

class LAS(nn.Module):
    def __init__(self, vocab_size, input_dim, embedding_dim,lis_hidden_dim,dec_hidden_size, context_dim,num_layers):
        super(LAS, self).__init__()
        self.Encoder = Listener(input_dim, lis_hidden_dim, value_size=128, key_size=128)
        self.Decoder = Decoder(vocab_size, embedding_dim, dec_hidden_size,context_dim,num_layers)

    def forward(self, speech_input, speech_lens,text,criterion=None,text_lens=None,tf_prob = 0,is_train = True):
        key, value,out_lens = self.Encoder(speech_input,speech_lens)
        if is_train:
          output = self.Decoder(text,text_lens,out_lens,criterion,key,value,tf_prob,is_train)
        else:
          output = self.Decoder.forward_test(text.float(),key,value,out_lens)

        return output