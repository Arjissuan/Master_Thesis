import torch
import torch.nn as nn
import torch.nn.functional as F


class PeptideCNN(nn.Module):
    def __init__(self, seq_len, num_classes=5, vocab_size=22, debug=False):
        super().__init__()
        self.debug = debug

        # Convolutional block
        # Input shape: (batch, 1, seq_len, vocab_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, vocab_size))
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len, vocab_size)
            dummy_out = self._forward_features(dummy)
            conv_output_size = dummy_out.view(1, -1).size(1)

        # Fully connected head

        self.fc1 = nn.Linear(conv_output_size, conv_output_size) # 172*32/2
        self.fc2 = nn.Linear(conv_output_size, num_classes)  # 172*32/2

        if self.debug:
            print(f"[DEBUG:init] seq_len={seq_len}, conv_output_size={conv_output_size}")

    def _forward_features(self, x):
        # x: (batch, 1, seq_len, vocab_size)
        x = self.conv1(x)       # -> (batch, 32, seq_len-2, 1)
        x = F.relu(x)
        x = x.squeeze(-1)       # remove vocab dimension -> (batch, 32, seq_len-2)
        x = self.pool(x)        # -> (batch, 32, (seq_len-2)//2)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        if self.debug:
            print(f"[DEBUG:forward] Feature map before flatten: {x.shape}")

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.debug:
            print(f"[DEBUG:forward] Output: {x.shape}")

        return x


class PeptideLinear(nn.Module):
    def __init__(self, seq_len, vocab_size=22, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(seq_len * vocab_size, num_classes)

    def forward(self, xb):
        # xb shape: (batch, 1, seq_len, vocab_size)
        xb = xb.view(xb.size(0), -1)  # flatten all except batch
        return self.fc(xb)

