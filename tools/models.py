import torch.nn as nn

class MyRegressor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MyRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x
    
    def forward(self, x):
        x = self.embedding(x)
        # x = self._pool(x)
        # print(x.shape)
        _, (h, _) = self.rnn(x)
        x = self.fc(h)
        return x