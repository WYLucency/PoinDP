from cProfile import label
import dgl
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import HeteDP
from data.utils import micro_macro_f1_score
import dataloads as dls

def evaluate(model, gs, features, labels, train_mask,val_mask,mask, score):
    model.eval()
    with torch.no_grad():
        if features == None:
            features = model.embed_layer()
        embed,_ = model.layer(gs, features)
        logits = model.predict(embed)
    return score(logits[train_mask], labels[train_mask]), score(logits[val_mask], labels[val_mask]), score(logits[mask], labels[mask]), embed

def add_noies(features, delta, sens_size, epsilon):
    sigma = torch.sqrt(2 * torch.log(1.25 / delta)) * sens_size / epsilon
    noise = sigma * torch.normal(mean=torch.zeros_like(features).data,std=torch.tensor(1.0).cuda())
    return features + alpha*20*noise


d_names=['dblp','imdb']
times=range(10)
wait_total=100
index_pip = 'PoinDP'
list_train_acc={index_pip:[i for i in times]}
list_valid_acc={index_pip:[i for i in times]}
list_test_acc={index_pip:[i for i in times]}
list_wf1={index_pip:[i for i in times]}
list_mf1={index_pip:[i for i in times]}
list_train_acc_sum={index_pip:0}
list_valid_acc_sum={index_pip:0}
list_test_acc_sum={index_pip:0}
list_wf1_sum={index_pip:0}
list_mf1_sum={index_pip:0}
lr = 0.005
num_hidden = 8
num_heads = 8
dropout = 0.8
weight_decay = 0.001

delta = torch.tensor(10e-5)
epsilon = 1.0
alpha = 0.01
score = micro_macro_f1_score
metrics = 'Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f} | Train Weighted-F1 {:.4f}' \
                ' | Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f} | Val Weighted-F1 {:.4f}'


for d_name in d_names:
    data=dls.loaddatas(d_name,d_name,0)
    g = data[0]
    f2=open('dp_scores/' +d_name+ '_scores.txt', 'w+')
    f2.write('{0:7} {1:7}\n'.format(d_name,index_pip))
    f2.write('{0:7} {1:7} {2:7} {3:7} {4:7} {5:7}\n'.format('run','train','valid','test_acc','w-f1','m-f1'))
    f2.flush()

    category = data.predict_ntype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for time in times:
        best_val_acc = test_acc = 0.0
        best_val_loss = np.inf
        num_classes = data.num_classes(category)
        degress = torch.zeros(g.num_nodes(category))
        gs = [dgl.metapath_reachable_graph(g, metapath) for metapath in data.metapaths(category)]
        for i in range(len(gs)):
            gs[i] = dgl.add_self_loop(dgl.remove_self_loop(gs[i]))
            degress +=  gs[i].in_degrees()   #in=out
            gs[i] = gs[i].to(device)
        
        p_values = dls.get_poincare(d_name)
        features = g.nodes[category].data['feat']
        model = HeteDP(
        len(gs), g.num_nodes(category), features.shape[1], num_hidden, num_classes, num_heads, p_values, p_values.shape[1], dropout)

        labels = g.nodes[category].data['label']
        labels = labels.to(device)
        train_mask, val_mask, test_mask = dls.get_mask(d_name, gs[0], 2)
            
        g = g.to(device)
        features = features.cuda()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        sens_size = 2.0
        sens_size = torch.tensor([sens_size]).expand(g.num_nodes(category))
        sens_size = sens_size.to(device)
        sens_size = sens_size.reshape(-1,1)
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            # train_features = features
            train_features = add_noies(features, delta, sens_size, epsilon)
            logits, atten, poinC = model(gs, train_features)
            sens_size = poinC.reshape(-1,1)
            # sens_size = torch.matmul(atten,sens_size)
            sens_size = sens_size.detach()
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            train_metrics,val_metrics,test_metrics, _= evaluate(model, gs, features, labels,train_mask,val_mask, test_mask, score)
            print('Tarin F1 {:.4f} {:.4f} {:.4f} | Valid F1 {:.4f} {:.4f} {:.4f} | Test F1 {:.4f} {:.4f} {:.4f}'.format(*test_metrics,*test_metrics,*test_metrics))
            if val_metrics[0]>=best_val_acc:
                test_acc=test_metrics[0]
                w_f1 = test_metrics[1]
                m_f1 = test_metrics[2]
                best_val_acc=val_metrics[0]
                train_re=train_metrics[0]
                wait_step=0
            else:
                wait_step += 1
                if wait_step == wait_total:
                    print('Early stop! Max accuracy: ', best_val_acc)
                    break
        del model

        list_train_acc[index_pip][time]=train_re
        list_valid_acc[index_pip][time]=best_val_acc
        list_test_acc[index_pip][time]=test_acc
        list_wf1[index_pip][time]=w_f1
        list_mf1[index_pip][time]=m_f1
        list_train_acc_sum[index_pip]=list_train_acc_sum[index_pip]+train_re/len(times)
        list_valid_acc_sum[index_pip]=list_valid_acc_sum[index_pip]+best_val_acc/len(times)
        list_test_acc_sum[index_pip]=list_test_acc_sum[index_pip]+test_acc/len(times)
        list_wf1_sum[index_pip]=list_wf1_sum[index_pip]+w_f1/len(times)
        list_mf1_sum[index_pip]=list_mf1_sum[index_pip]+m_f1/len(times)

        print(metrics.format(epoch, loss.item(), *train_metrics, *val_metrics, *test_metrics))
        del logits, train_features
        
        f2.write('{0:4d} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n'.format(time,list_train_acc[index_pip][time],list_valid_acc[index_pip][time],list_test_acc[index_pip][time],list_wf1[index_pip][time],list_mf1[index_pip][time]))
        f2.flush()
        g = g.cpu()
        del sens_size, features, degress, labels, train_mask, test_mask, val_mask, gs
    f2.write('{0:4} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n'.format('std',np.std(list_train_acc[index_pip]),np.std(list_valid_acc[index_pip]),np.std(list_test_acc[index_pip]),np.std(list_wf1[index_pip]),np.std(list_mf1[index_pip])))
    f2.write('{0:4} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n'.format('mean',np.mean(list_train_acc[index_pip]),np.mean(list_valid_acc[index_pip]),np.mean(list_test_acc[index_pip]),np.mean(list_wf1[index_pip]),np.mean(list_mf1[index_pip])))
    f2.close()


