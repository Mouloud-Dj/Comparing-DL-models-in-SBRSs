from tqdm import tqdm
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from torch_geometric.nn import GCNConv,global_add_pool,global_mean_pool
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output
    
    def fit(self, data, device='cpu', num_epochs=30):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0

            for batch, (X, y) in tqdm(enumerate(data)):
                # Move data to the appropriate device
                X, y = X.to(device), y.to(device)

                # Forward pass
                predictions = self(X)
                loss = loss_function(predictions, y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(data)}')
    def test(self,data,device="cpu"):
        self.eval()
        correct = 0
        rr=0
        with torch.no_grad():
            for X, y in tqdm(data):
                X, y = X.to(device), y.to(device)
                predictions = self(X)[0]
                top_probs, top_indices = torch.topk(predictions, 10, largest=True, sorted=True)
                top_indices = top_indices.tolist()
                if(y[0] in top_indices):
                    rr+=1/(top_indices.index(y[0])+1)
                    correct+=1
        accuracy = correct / len(data)
        mrr=rr/len(data)
        print(f"Test P@20: {accuracy * 100:.2f}% MRR@20 {mrr*100:.2f}")

class Bi_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional=True):
        super(Bi_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        
        # Adjust the output dimension if using bidirectional LSTM
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden.squeeze(0)
        
        out = self.fc(hidden)
        return out
    
    def fit(self, data, device='cpu', num_epochs=30):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0

            for batch, (X, y) in tqdm(enumerate(data)):
                # Move data to the appropriate device
                X, y = X.to(device), y.to(device)

                # Forward pass
                predictions = self(X)
                loss = loss_function(predictions, y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(data)}')
    def test(self,data,device="cpu"):
        self.eval()
        correct = 0
        rr=0
        with torch.no_grad():
            for X, y in tqdm(data):
                X, y = X.to(device), y.to(device)
                predictions = self(X)[0]
                top_probs, top_indices = torch.topk(predictions, 10, largest=True, sorted=True)
                top_indices = top_indices.tolist()
                if(y[0] in top_indices):
                    rr+=1/(top_indices.index(y[0])+1)
                    correct+=1
        accuracy = correct / len(data)
        mrr=rr/len(data)
        print(f"Test P@20: {accuracy * 100:.2f}% MRR@20 {mrr*100:.2f}")


class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    def fit(self,data,device="cpu",epoch=30):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epoch):
            all_loss=0
            print("train:")
            for (inputs, labels) in tqdm(data):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = self(inputs)
                loss = loss_function(output, labels)
                all_loss+=loss
                loss.backward()
                optimizer.step()

            print(f"epoch:{epoch+1}, loss:{all_loss/len(data)} %")
    def test(self,X_test,Y_test,device="cpu"):
        with torch.no_grad():
              correct = 0
              total = 0
              rr=0
              for x, y in tqdm(zip(X_test, Y_test)):
                  target=torch.tensor(y).to(device)
                  output = self(x.to(device))
                  top_probs, top_indices = torch.topk(output, 20, largest=True, sorted=True)
                  top_indices = top_indices.tolist()
                  total += 1
                  if(target in top_indices):
                    rr+=1/(top_indices.index(target)+1)
                    correct+=1
        accuracy = correct / total
        mrr=rr/total
        print(correct)
        print(total)
        print(f"Test P@20: {accuracy * 100:.2f}% MRR@20 {mrr*100:.2f}")

class VanillaGNNLayer(torch.nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.linear = Linear(dim_in, dim_out, bias=False)
  def forward(self, x, adjacency):
    x = self.linear(x)
    x = torch.sparse.mm(adjacency, x)
    return x.cuda()
class GNN(torch.nn.Module):
    def __init__(self,vocab_size, num_nodes,embedding_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.GNN1 = GCNConv(embedding_dim*num_nodes, hidden_dim)
        self.GNN2 = GCNConv(hidden_dim, hidden_dim)
        self.fc2= nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, data):
        x, adj_t = data.x, data.edge_index
        x=self.embedding(x)
        x1= self.GNN1(x.view(x.size(0),-1), adj_t)
        x1 = F.relu(x1)

        x1= self.GNN2(x1, adj_t)
        x1 = F.relu(x1)
        x1 = global_add_pool(x1, data.batch)
        # Fully connected layer
        x=  self.fc3(x1)
        return x
    def fit(self,data,device="cpu",epochs=30):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            all_loss=0
            i=0
            print("train:")
            for j in tqdm(data):
                optimizer.zero_grad()
                output = self(j.to(device))
                target = j.y.to(device)
                loss = loss_function(output, target)
                all_loss+=loss
                loss.backward()
                optimizer.step()
                i+=1
            print(f"epoch:{epoch+1}, loss:{all_loss/len(data)} %")
    def test(self,data,device="cpu"):
        with torch.no_grad():
              correct = 0
              rr=0
              for i in tqdm(data):
                  target=i.y.to(device)
                  output = self(i.to(device))[0]
                  top_probs, top_indices = torch.topk(output, 20, largest=True, sorted=True)
                  top_indices = top_indices.tolist()
                  if(target in top_indices):
                    correct+=1
                    rr+=1/(top_indices.index(target)+1)
        accuracy = correct / len(data)
        mrr=rr/len(data)
        print(f"{accuracy * 100:.2f}% MRR@20: {mrr *100:.2f}")