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

from utils import WrappedNormal
from poincare import PoincareBallWrapped


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


def get_sigma(epsilon, poin_w, sensitivity=1, alpha=1, delta = 10e-5):
    sigma = torch.sqrt((sensitivity**2 * alpha * poin_w / (2 * epsilon)).clone().detach())
    # sigma = torch.sqrt(2 * torch.log(1.25 / torch.tensor(delta))) * poin_w / epsilon
    return sigma

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

        # self.angle = False


        self.lin = Linear(in_channels, out_channels, bias=False)

        widths_p=[n_components_p,out_channels]
        widths_a=[1,out_channels]
        self.w_mlp_out_p=create_wmlp(widths_p,out_channels,1)
        self.w_angle=create_wmlp(widths_a,out_channels,1)
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

    def forward(self, x: Tensor, edge_index: Adj, angle = None,
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
        if angle:
            intra_w = torch.angle(self.w_mul_p)

        p_weight=self.w_mlp_out_p(self.w_mul_p + intra_w) if angle else self.w_mlp_out_p(self.w_mul_p)
        p_weight=F.leaky_relu(p_weight)

        if self.bias is not None:
            out += self.bias

        # out = add_noise(out)
        return out, p_weight

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out
        
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class GCN_Net(torch.nn.Module):
    def __init__(self,data,num_features,num_hidden,num_classes,w_mul_p,n_components_p,epsilon,delta):
        super(GCN_Net, self).__init__()
        self.p_w = w_mul_p
        # self.trans = nn.Linear(num_features, num_features//2)
        self.h_inter = nn.Sequential(
            GCNConv(num_features, num_hidden,w_mul_p,n_components_p, cached=True), 
            GCNConv(num_hidden, num_classes,w_mul_p,n_components_p, cached=True)
        )
        self.h_intra = nn.Sequential(
            GCNConv(num_features, num_hidden,w_mul_p,n_components_p, cached=True), 
            GCNConv(num_hidden, num_classes,w_mul_p,n_components_p, cached=True)
        )
        hidden_dim = 128
        self.allocate = nn.Sequential(
            nn.Linear(num_hidden+num_classes, hidden_dim), 
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.atten = nn.Sequential(
            nn.Linear(num_classes, hidden_dim), 
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.chage=nn.Linear(num_classes,2)
        self.epsilon = epsilon
        self.delta = delta
        self.wrapped_nd = WrappedNormal(torch.tensor([0.]).cuda(), torch.tensor([1.]).cuda(), PoincareBallWrapped(c=torch.Tensor([1.]).cuda()))
        

    def forward(self,data,features=None,perturb=None):
        if features is not None:
            x = features
        else:
            x = data.x

        '''PoinDP'''
        angle = False
        weight = []
        out = []
        for i in [self.h_inter, self.h_intra]:
            h,weight1 = i[0](x, data.edge_index, angle)
            h = F.elu(h)
            h = F.dropout(h,p=0.6,training=self.training)
            h,weight2 = i[1](h, data.edge_index, angle)
            out.append(h)
            # weight.append(weight2)
            weight.append(torch.hstack((weight1, weight2)))
            angle = True


        '''PERTURBATION'''
        if perturb:
            
            weight = torch.stack(weight, dim=1)
            w = F.leaky_relu(self.allocate(weight))
            gama = torch.softmax(w.mean(dim=0), dim=0)

            out_n = []
            for i in range(2):
                aware = weight[:,i].sum(dim=1)
                sigma = get_sigma(gama[i] * self.epsilon,aware).reshape(-1,1).expand_as(out[i])#mlp_weight
                temp_out = out[i]
                noise = sigma * self.wrapped_nd.sample(temp_out.size(), torch.tensor([1.]).cuda()).squeeze() 
                out_n.append(temp_out+noise)
        else:
            noise = 1
            out_n = out

        return F.log_softmax(torch.stack(out_n, dim=1).sum(dim=1), dim=1), noise


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data,name,num_features,num_classes,epsilon,delta):
    data.edge_index, _ = remove_self_loops(data.edge_index)
    keys=np.load('data/poincare_weight/'+name+'_keys.npy')
    values=np.load('data/poincare_weight/'+name+'_values.npy')
    w_mul_p = values
    w_mul_p = dict(zip(keys, values))
    data.n_components_p = 2
    alls = dict(enumerate(np.ones((data.num_nodes,data.n_components_p)), 0))
    alls.update(w_mul_p)
    w_mul_p = torch.tensor([alls[i] for i in alls])
    data.w_mul_p = w_mul_p.to(torch.float32)
    data.edge_index, _ = add_self_loops(data.edge_index,num_nodes=data.x.size(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data.w_mul_p = data.w_mul_p.to(device)
    data = data.to(device)
    model= GCN_Net(data,num_features,16,num_classes,data.w_mul_p,data.n_components_p,epsilon,delta).to(device)
    return model, data


def create_wmlp(widths,nfeato,lbias):
    mlp_modules=[]
    for k in range(len(widths)-1):
        mlp_modules.append(Linear(widths[k],widths[k+1],bias=False))
        mlp_modules.append(LeakyReLU(0.2,True))
    mlp_modules.append(Linear(widths[len(widths)-1],nfeato,bias=lbias))
    return seq(*mlp_modules)