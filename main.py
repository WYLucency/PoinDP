import torch
import dataloads as dls
import torch.nn.functional as F
import numpy as np
from model import call
from sklearn.metrics import f1_score


def train(train_mask,data,features,perturb=None):
    model.train()
    optimizer.zero_grad()
    logits, noise = model(data,features=features,perturb=perturb)
    loss = F.nll_loss(logits[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return noise,loss.item()

def test(train_mask,val_mask,test_mask):
    with torch.no_grad():
        model.eval()
        (logits, noise), accs = model(data, perturb=True), []
        for mask in [train_mask,val_mask,test_mask]:
            pred = logits[mask].max(1)[1]
            acc = f1_score(pred.cpu(), data.y[mask].cpu(), average='micro')
            accs.append(acc)
        
        for_loss,_ = model(data, perturb=True)
        accs.append(F.nll_loss(for_loss[val_mask], data.y[val_mask]))
        accs.append(f1_score(pred.cpu(), data.y[mask].cpu(), average='weighted'))
        accs.append(f1_score(pred.cpu(), data.y[mask].cpu(), average='macro'))
        accs.append(noise)
        accs.append(logits)
    return accs


#DP settings
alpha = 0.01
delta = torch.tensor(10e-5)
epsilons = [1]
# epsilons = [0.01,0.05,0.1,0.5,1]

#load dataset
times=range(5)
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
d_names=['Cora','Computers','Photo','PubMed','Citeseer']

for d_name in d_names:
    for epsilon in epsilons:
        f2=open('dp_scores/' +d_name+ str(epsilon)+'_cores.txt', 'w+')
        f2.write('{0:7} {1:7}\n'.format(d_name,index_pip))
        f2.write('{0:7} {1:7} {2:7} {3:7} {4:7} {5:7}\n'.format('run','train','valid','test_acc','w-f1','m-f1'))
        f2.flush()
        if d_name=='Cora' or d_name=='Citeseer' or d_name=='PubMed':
            d_loader='Planetoid'
        # elif d_name=='Amazon':
        elif d_name=='Computers' or d_name=='Photo':
            d_loader='Amazon'
        elif d_name=='Physics':
            d_loader='Coauthor'
        else:
            d_loader=d_name
        
        dataset=dls.loaddatas(d_loader,d_name,0)
        for time in times:
            weight = 2.0
            data=dataset[0]
            train_mask=data.train_mask.bool()
            val_mask=data.val_mask.bool()
            test_mask=data.test_mask.bool()

            model,data = call(data,dataset.name,data.x.size(1),dataset.num_classes,epsilon,delta)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
            best_val_acc = test_acc = 0.0
            best_val_loss = np.inf
            losses = []
            noises = 0
            for epoch in range(1, 201): 
                features=None
                noise,loss = train(train_mask, data, features, perturb=True)
                
                losses.append(loss)
                train_acc,val_acc,tmp_test_acc,val_loss,tmp_w_f1,tmp_m_f1,_,result= test(train_mask,val_mask,test_mask)
                # noises += noise
                noises += torch.abs(noise)
                print("acc:", train_acc,val_acc,tmp_test_acc,val_loss.item(),tmp_w_f1,tmp_m_f1)
                if val_acc>=best_val_acc:
                    test_acc=tmp_test_acc
                    w_f1 = tmp_w_f1
                    m_f1 = tmp_m_f1
                    best_val_acc=val_acc
                    train_re=train_acc
                    best_val_loss=val_loss
                    wait_step=0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                        break
            del model
            del data
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
            log ='Epoch: 200, dataset name: '+ d_name + ', Method: '+ index_pip + ' Test: {0:.4f} {1:.4f} {2:.4f}\n'
            print((log.format(list_test_acc[index_pip][time],list_wf1[index_pip][time],list_mf1[index_pip][time])))
            f2.write('{0:4d} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n'.format(time,list_train_acc[index_pip][time],list_valid_acc[index_pip][time],list_test_acc[index_pip][time],list_wf1[index_pip][time],list_mf1[index_pip][time]))
            f2.flush()
        f2.write('{0:4} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n'.format('std',np.std(list_train_acc[index_pip]),np.std(list_valid_acc[index_pip]),np.std(list_test_acc[index_pip]),np.std(list_wf1[index_pip]),np.std(list_mf1[index_pip])))
        f2.write('{0:4} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n'.format('mean',np.mean(list_train_acc[index_pip]),np.mean(list_valid_acc[index_pip]),np.mean(list_test_acc[index_pip]),np.mean(list_wf1[index_pip]),np.mean(list_mf1[index_pip])))
        f2.close()
