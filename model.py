from tkinter.messagebox import NO
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.nn import Sequential as seq, Parameter,LeakyReLU,init,Linear
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax

from dgl.nn.pytorch import GATConv

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 w_mul_p,n_components_p,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.w_mul_p = w_mul_p

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False)

        widths_p=[n_components_p,out_channels]
        self.w_mlp_out_p=create_wmlp(widths_p,out_channels,1)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
            edge_weight = edge_weight.view(-1, 1)
        
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        p_weight=self.w_mlp_out_p(self.w_mul_p)
        p_weight=F.leaky_relu(p_weight)

        if self.bias is not None:
            out += self.bias

        return out, p_weight

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out
        
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class GCN_Net(torch.nn.Module):
    def __init__(self,data,num_features,num_hidden,num_classes,w_mul_p,n_components_p):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden,w_mul_p,n_components_p, cached=True)
        self.conv2 = GCNConv(num_hidden, num_classes,w_mul_p,n_components_p, cached=True)

    def forward(self,data,features=None):
        if features is not None:
            x = features
        else:
            x = data.x
        x,weight1 = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=0.6,training=self.training)
        x,weight2 = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1), torch.norm(torch.hstack((weight1, weight2)),dim=1)

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes):
    data.edge_index, _ = remove_self_loops(data.edge_index)
    keys=np.load('data/poincare_weight/'+name+'_keys.npy')
    values=np.load('data/poincare_weight/'+name+'_values.npy')
    w_mul_p = dict(zip(keys, values))
    data.n_components_p = values.shape[1]
    alls = dict(enumerate(np.ones((data.num_nodes,data.n_components_p)), 0))
    alls.update(w_mul_p)
    w_mul_p = torch.tensor([alls[i] for i in alls])
    data.w_mul_p = w_mul_p.to(torch.float32)
    data.edge_index, _ = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data.w_mul_p = data.w_mul_p.to(device)
    data = data.to(device)
    model= GCN_Net(data,num_features,16,num_classes,data.w_mul_p,data.n_components_p).to(device)
    return model, data


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 num_node,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.num_node = num_node
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.ParameterDict()
        embed = nn.Parameter(torch.Tensor(self.num_node, self.embed_size))
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
        self.embeds = embed

    def forward(self, block=None):
        """Forward computation"""
        return self.embeds

class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z, alpha, gs):
        atten = 0
        w = self.project(z)
        w = F.leaky_relu(w).mean(dim=0)
        beta = torch.softmax(w, dim=0)
        for g,a,b in zip(gs, alpha, beta):
            src, dst = g.edges()
            indices = np.vstack((src.cpu(), dst.cpu()))
            values = alpha[a] * b
            edge_g = torch.sparse_coo_tensor(indices, values, (g.num_nodes(),g.num_nodes()))
            atten += edge_g.to_dense()

        beta = beta.expand((z.shape[0],) + beta.shape)
        z = (beta * z).sum(dim=1)
        return z, atten


class HeteLayer(nn.Module):
    def __init__(self, num_metapaths, in_dim, out_dim, num_heads, dropout):
        super().__init__()
        self.gats = nn.ModuleList([
            GATConv(in_dim, out_dim, num_heads, dropout, dropout, activation=F.elu)
            for _ in range(num_metapaths)
        ])
        self.semantic_attention = SemanticAttention(in_dim=num_heads * out_dim)

    def forward(self, gs, h):
        zp = []
        alpha ={}
        for i, gat, g in zip(range(len(gs)),self.gats, gs):
            temp_zp, temp_alpha = gat(g, h, True)
            zp.append(temp_zp.flatten(start_dim=1))
            temp_alpha = temp_alpha.flatten(start_dim=1).mean(1)
            alpha[i]=temp_alpha

        zp = torch.stack(zp, dim=1)
        z, atten = self.semantic_attention(zp, alpha, gs) 
        return z, atten


class HeteDP(nn.Module):
    def __init__(self, num_metapaths, num_node, in_dim, hidden_dim, out_dim, num_heads, p_values, num_p, dropout):
        super().__init__()
        self.w_mul_p = p_values
        widths_p=[num_p,out_dim]
        self.w_mlp_out_p=create_wmlp(widths_p,out_dim,1)
        self.embed_layer = RelGraphEmbed(num_node, in_dim)
        self.layer = HeteLayer(num_metapaths, in_dim, hidden_dim, num_heads, dropout)
        self.predict = nn.Linear(num_heads * hidden_dim, out_dim)
    
    def forward(self, gs, h):      
        h, atten = self.layer(gs, h)  
        out = self.predict(h) 
        p_weight=self.w_mlp_out_p(self.w_mul_p)
        p_weight=F.leaky_relu(p_weight)
        return out, atten, torch.norm(p_weight,dim=1)

def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)