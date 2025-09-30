import torch
import torch.nn as nn

class PeptideCNN(nn.Module):
    def __init__(self, seq_len, vocab_size=22, num_classes=5, num_filters=32, kernel_size=3):
        super().__init__()
        # input: (batch, 1, seq_len, vocab_size)
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(kernel_size, vocab_size)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters * (seq_len - kernel_size + 1), num_classes)
        )

    def forward(self, xb):
        # xb shape: (batch, 1, seq_len, vocab_size)
        return self.conv_net(xb)

class PeptideLinear(nn.Module):
    def __init__(self, seq_len, vocab_size=22, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(seq_len * vocab_size, num_classes)

    def forward(self, xb):
        # xb shape: (batch, 1, seq_len, vocab_size)
        xb = xb.view(xb.size(0), -1)  # flatten all except batch
        return self.fc(xb)

