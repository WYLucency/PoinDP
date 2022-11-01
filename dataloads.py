import torch_geometric.datasets as dt
import torch_geometric.transforms as T
import torch
import numpy as np
from dgl.data.utils import generate_mask_tensor, idx2mask

from data import dblp,imdb,utils as ut

def loaddatas(d_loader,d_name,spcial):
    if d_loader=='Planetoid':
        dataset = dt.Planetoid(root='data/'+d_name, name=d_name, transform=T.NormalizeFeatures())
    elif d_loader=='dblp':
        dataset = dblp.DBLPHeCoDataset()
        return dataset
    elif d_loader == "imdb":
        dataset = imdb.IMDbDataset()  
        return dataset

    data = get_split(dataset, spcial)
    dataset.data = data
    return dataset

def get_split(dataset, spcial):
    data = dataset.data
    keys=np.load('data/poincare_weight/'+dataset.name+'_keys.npy')
    values=np.load('data/poincare_weight/'+dataset.name+'_values.npy')
    w_mul_p = dict(zip(keys, values))
    data.n_components_p = values.shape[1]
    alls = dict(enumerate(np.ones((data.num_nodes,data.n_components_p)), 0))
    alls.update(w_mul_p)
    values = np.array([alls[i] for i in alls])
    sorted, indices = torch.sort(torch.norm(torch.tensor(values),dim=1),descending=True)
    #split ratio 1:1:8
    if spcial == 1:#(0,0.33)
        train_idx, val_idx, test_idx = ut.split_idx1(indices[:data.num_nodes//3],indices[data.num_nodes//3:], 0.3, 0.1, 42)
    elif spcial == 2:#(0.66,1)
        train_idx, val_idx, test_idx = ut.split_idx1(indices[data.num_nodes//3+data.num_nodes//3:],indices[:data.num_nodes//3+data.num_nodes//3], 0.3, 0.1, 42)
    else:#random
        train_idx, val_idx, test_idx = ut.split_idx(np.arange(data.num_nodes), 0.1, 0.1, 42)

    data.train_mask = generate_mask_tensor(idx2mask(train_idx, data.num_nodes))
    data.val_mask = generate_mask_tensor(idx2mask(val_idx, data.num_nodes))
    data.test_mask = generate_mask_tensor(idx2mask(test_idx, data.num_nodes))
    return data

def get_poincare(name):
    values=np.load('data/poincare_weight/'+name+'_values.npy')
    w_mul_p = torch.tensor(values).to(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return w_mul_p.to(device)

def get_mask(name, g, spcial):
    values=np.load('data/poincare_weight/'+name+'_values.npy')
    sorted, indices = torch.sort(torch.norm(torch.tensor(values),dim=1),descending=True)
    if spcial == 1:
        train_idx, val_idx, test_idx = ut.split_idx1(indices[:g.num_nodes()//3],indices[g.num_nodes()//3:], 0.3, 0.1)
    elif spcial == 2:
        train_idx, val_idx, test_idx = ut.split_idx1(indices[g.num_nodes()//3+g.num_nodes()//3:],indices[:g.num_nodes()//3+g.num_nodes()//3], 0.3, 0.1)
    else:
        train_idx, val_idx, test_idx = ut.split_idx(np.arange(g.num_nodes()), 0.1, 0.1, 42)

    train_mask = generate_mask_tensor(idx2mask(train_idx, g.num_nodes()))
    val_mask = generate_mask_tensor(idx2mask(val_idx, g.num_nodes()))
    test_mask = generate_mask_tensor(idx2mask(test_idx, g.num_nodes()))
    return train_mask, val_mask, test_mask