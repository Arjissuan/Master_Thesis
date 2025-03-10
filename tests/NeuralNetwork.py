import torch
from torch.nn.modules.module import register_module_forward_hook


class NeuralNetwork(torch.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Module.register_module("fc1", )
        self.fc2 = torch.nn.Module.register_module("fc2")
        self.fc3 = torch.nn.Module.register_module("fc3")


    def forward(self, x):
        x = torch.relu(self.fc1(x))