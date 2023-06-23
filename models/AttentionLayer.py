import math
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn


######################################################################
# Multi-Head Attention
######################################################################
class MHAtt(nn.Module):
    """普通多头注意力机制
    """
    def __init__(self, input_channels, h=4, q_dim=None, v_dim=None):
        # q_dim 和 v_dim 正常情况下相等
        super(MHAtt,self).__init__()
        self.head = h
        # 确定进行注意力的数据特征维度是先增大特征维度再切分，还是在原维度上切分
        q_dim = q_dim or (input_channels // h)
        v_dim = v_dim or (input_channels // h)
        # Query, keys and value matrices
        self._W_q = nn.Linear(input_channels, q_dim*h)  # q,k必须维度相同，不然无法做点积
        self._W_k = nn.Linear(input_channels, q_dim*h)
        self._W_v = nn.Linear(input_channels, v_dim*h)
        # Output linear function
        self._W_o = nn.Linear(h*v_dim, input_channels)

    def forward(self, x):
        """
        [B, C, N, T]
        """
        x = x.permute(0, 2, 3, 1)
        queries = torch.cat(self._W_q(x).chunk(self.head, dim=-1), dim=0)
        keys = torch.cat(self._W_k(x).chunk(self.head, dim=-1), dim=0)
        values = torch.cat(self._W_v(x).chunk(self.head, dim=-1), dim=0)
        # Scaled Dot Product
        D = queries.size(2)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / np.sqrt(D)

        # 对最后一个维度(v)做softmax
        attn_score = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn_score, values)
        # Concatenat the heads
        attention_heads = torch.cat(context.chunk(self.head, dim=0), dim=-1)

        # Apply linear transformation W^O
        out = self._W_o(attention_heads)
        # 返回经过多头注意力后的数据及注意力分数(后者可用于可视化分析)
        return out.permute(0, 3, 1, 2)


######################################################################
# Informer encoder layer
######################################################################
def PositionalEmbedding(input_windows, feature_dims):
    """构建时间位置编码
    Parameters
    ----------
    input_windows:
        Time window length, i.e. input_windows.
    feature_dims:
        Dimension of the model vector.
    Returns
    -------
        Tensor of shape (input_windows, feature_dims).
    """
    TPE = torch.zeros((input_windows, feature_dims))  # (input_windows, feature_dims)
    pos = torch.arange(input_windows).unsqueeze(1)  # (input_windows, 1)
    TPE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, feature_dims, 2, dtype=torch.float32)/feature_dims))
    TPE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, feature_dims, 2, dtype=torch.float32)/feature_dims))
    return TPE


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # kernel_size = 3
        # pad = (kernel_size - 1) // 2
        # self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.value_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class InformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x * math.sqrt(self.d_model)
        x = x + PositionalEmbedding(T, C).to(x.device)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# Transformer encoder layer
######################################################################
class TriangularCausalMask():
    def __init__(self, B, L, device):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(TransformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        x = x + PositionalEmbedding(T, C).to(x.device)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output

