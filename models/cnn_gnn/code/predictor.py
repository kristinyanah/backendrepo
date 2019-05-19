import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(object):
    def __init__(self, model, setting):
        """Load the pre-trained model from the directory of output/model."""
        self.model = model
        model.load_state_dict(torch.load('../output/model/' + setting))

    def predict(self, dataset, smiles_sequence_list):
        z_list, t_list = [], []
        for data in dataset:
            z = self.model(data)
            z_list.append(z[1])
            t_list.append(np.argmax(z))
        with open('prediction_results.txt', 'w') as f:
            f.write('smiles sequence '
                    'interaction_probability binary_class\n')
            for (c, p), z, t in zip(smiles_sequence_list, z_list, t_list):
                f.write(' '.join(map(str, [c, p, z, t])) + '\n')

        c = smiles_sequence_list[0]
        p = smiles_sequence_list[0]
        z = z_list[0]
        t = t_list[0]
        return c,p,z,t


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint=None, dim=None, layer_gnn=None, window=None,
     layer_cnn=None, n_word=None):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.n_fingerprint = n_fingerprint
        self.dim = dim
        self.layer_cnn = layer_cnn
        self.layer_gnn = layer_gnn
        self.window = window
        self.n_word = n_word


        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def cnn(self, xs, i):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        hs = torch.relu(self.W_cnn[i](xs))
        return torch.squeeze(torch.squeeze(hs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        for i in range(layer):
            hs = self.cnn(xs, i)
            x = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(hs))
            weights = torch.tanh(F.linear(x, hs))
            xs = torch.t(weights) * hs
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def __call__(self, data):

        fingerprints, adjacency, words = data

        x_fingerprints = self.embed_fingerprint(fingerprints)
        x_compound = self.gnn(x_fingerprints, adjacency, self.layer_gnn)

        x_words = self.embed_word(words)
        x_protein = self.attention_cnn(x_compound, x_words, self.layer_cnn)

        y_cat = torch.cat((x_compound, x_protein), 1)
        z_interaction = self.W_out(y_cat)
        z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()

        return z