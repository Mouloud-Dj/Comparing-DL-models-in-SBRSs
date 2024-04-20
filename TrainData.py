from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm
import numpy as np
import YCData
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return sequences_padded, labels
def data_rnn(sequences, labels):
    sequences = [torch.tensor(s, dtype=torch.int64) for s in sequences]
    labels = torch.tensor(labels, dtype=torch.int64)
    padded_sequences = pad_sequence(sequences, batch_first=True)

    dataset = TensorDataset(padded_sequences, labels)

    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.Subset(dataset, range(train_size)),torch.utils.data.Subset(dataset, range(train_size, train_size + len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return train_loader,val_loader

def XforMLP(seq,items):
    x=torch.FloatTensor([0 for i in range(items)])
    j=1
    for i in range(len(seq)):
        x[seq[i]]=j
        j+=1
    return x

def data_mlp(sequences, labels):
    XMLP=list()
    items=max(labels)+1
    for i in tqdm(range(len(sequences))):
        XMLP.append(XforMLP(sequences[i],items))
    X_train=torch.FloatTensor(np.array(XMLP[:round(len(XMLP)*0.8)]))
    Y_train=torch.LongTensor(labels[:round(len(XMLP)*0.8)])
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32)
    x_test,y_test=XMLP[round(len(XMLP)*0.8):],labels[round(len(XMLP)*0.8):]
    return train_loader,x_test,y_test

def MakeEdges(seq):
    current=[[],[]]
    for i in range(len(seq)-1):
        current[0].append(seq[i])
        current[1].append(seq[i+1])
    return current
def GetFetEdge(seq,num_items):
    edges=MakeEdges(seq)
    new_features=torch.FloatTensor([[0] for i in range(num_items+1)])
    j=1
    for i in range(len(seq)):
        new_features[seq[i]]=j
        j+=1
    return new_features,edges
from tqdm import tqdm
from torch_geometric.data import Data
def make_graphs(X,Y):
    graphs=list()
    num_items=max(Y)+1
    for i in tqdm(range(len(X))):
        feture,edge=GetFetEdge(X[i],num_items)
        data=Data(x=feture.int(),edge_index=torch.tensor(edge), y=torch.tensor(Y[i]).long())
        graphs.append(data)
    return graphs
from torch_geometric.loader import DataLoader

def data_gnn(sequences, labels):
    g=make_graphs(sequences,labels)
    loader = DataLoader(g[:round(len(g)*0.8)], batch_size=32)
    test = g[round(len(g)*0.8):]
    return loader , test
