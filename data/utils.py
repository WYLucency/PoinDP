import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def split_idx(samples, train_size, val_size, random_state=None):
    train, val = train_test_split(samples, train_size=train_size, random_state=random_state)
    if isinstance(val_size, float):
        val_size *= len(samples) / len(val)
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test
    
def split_idx1(samples1, samples2, train_size, val_size, random_state=None):
    train, val = train_test_split(samples1, train_size=train_size, random_state=random_state)
    val = torch.cat((val,samples2))
    val, test = train_test_split(val, train_size=val_size, random_state=random_state)
    return train, val, test

def micro_macro_f1_score(logits, labels):
    """计算Micro-F1和Macro-F1得分

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float, float Micro-F1和Macro-F1得分
    """
    prediction = torch.argmax(logits, dim=1).cpu().long().numpy()
    labels = labels.cpu().numpy() 
    micro_f1 = f1_score(labels, prediction, average='micro')
    weighted_f1 = f1_score(labels, prediction, average='weighted')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, weighted_f1, macro_f1
