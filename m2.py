import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


def train(model, criterion, optimizer, input_seq, target_seq):
    model.train()
    optimizer.zero_grad()
    output_seq = model(input_seq)
    loss = criterion(output_seq, target_seq)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    counts = []
    bad_counts = []
    for _ in tqdm(range(100)):
        input_size = 1
        hidden_size = 64
        output_size = 1
        learning_rate = 0.01

        @torch.no_grad()
        def black_box_function(x):
            # Replace this with your own implementation
            return (x - 25) ** 2 + 17

        model = RNN(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        bad_count = 0
        mse = float("inf")
        while mse > 0.05:
            input_seq = torch.randn(1, 1, input_size)
            target_seq = black_box_function(model(input_seq))
            loss = train(model, criterion, optimizer, input_seq, target_seq)
            mse = loss / input_seq.size(0)
            bad_count += 1

        count = 0
        mse = float("inf")
        while mse > 0.5:
            input_seq = torch.randn(1, 1, input_size)
            target_seq = black_box_function(model(input_seq))
            loss = train(model, criterion, optimizer, input_seq, target_seq)
            mse = loss / input_seq.size(0)
            count += 1
        counts.append(count)
        bad_counts.append(bad_count)

    print(sum(counts) / len(counts))
    print(sum(bad_counts) / len(bad_counts))


if __name__ == "__main__":
    main()