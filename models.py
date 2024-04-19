from tqdm import tqdm
import torch
from torch import nn


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
    
    def fit(self, data, d='cpu', num_epochs=30):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0

            for batch, (X, y) in tqdm(enumerate(data)):
                # Move data to the appropriate device
                X, y = X.to(d), y.to(d)

                # Forward pass
                predictions = self(X)
                loss = loss_function(predictions, y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(data)}')
