import torch.nn as nn

class Peptide_CNN(nn.Module):
    def __init__(self, seq_len, num_filters:int = 32, kernel_size:int=3):
        super.__init__()
        self.seq_len = seq_len
        self.conv_net = nn.Sequential(
            nn.Conv2d(21, num_filters, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters*(seq_len-kernel_size+1), 1)
        )

    def forward(self, xb):
        xb = xb.permute(0,2,1)
        out = self.conv_net(xb)
        return out

class DNA_Linear(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        # the 4 is for our one-hot encoded vector length 4!
        self.lin = nn.Linear(4*seq_len, 1)

    def forward(self, xb):
        # reshape to flatten sequence dimension
        xb = xb.view(xb.shape[0],self.seq_len*4)
        # Linear wraps up the weights/bias dot product operations
        out = self.lin(xb)
        return out