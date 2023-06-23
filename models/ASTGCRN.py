import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .AttentionLayer import MHAtt, TransformerLayer, InformerLayer


class AGCN(nn.Module):
    """
    Adaptive Graph Convolution
    """
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x: [B, N, dim_in]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class GCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(GCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: (B, num_nodes, input_dim)
        # state: (B, num_nodes, hidden_dim)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class GCRN(nn.Module):
    """
    Graph Convolutional Recurrent Network
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers):
        super(GCRN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # x: (B, T, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class ASTGCRN(nn.Module):
    def __init__(self, args):
        super(ASTGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.alpha = nn.Parameter(torch.randn(args.cheb_k, 2).to(args.device), requires_grad=True)
        self.encoder = GCRN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        # Three modules based on self-attentive mechanisms
        # self.att = MHAtt(args.rnn_units)
        self.att = TransformerLayer(args.rnn_units)
        # self.att = InformerLayer(args.rnn_units)
        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv_1 = nn.Conv2d(in_channels=args.rnn_units,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=512,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden

        # output = output[:, -1:, :, :]  #B, 1, N, hidden
        # output = self.end_conv((output))  #B, T*C, N, 1
        # output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        # output = output.permute(0, 1, 3, 2)  #B, T, N, C

        output = self.att(output.permute(0, 3, 2, 1))  # [B, C, N, T]
        output = F.relu(output)
        output = F.relu(self.end_conv_1(output))
        output = self.end_conv_2(output)  # [B, 1, N, T]
        output = output.permute(0, 3, 2, 1)
        # print("output.shape:", output.shape)
        return output
