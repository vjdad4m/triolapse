import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = x + out
        out = self.elu(out)
        return out

class PolicyHead(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(2)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * grid_size * grid_size, grid_size * grid_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class ValueHead(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(grid_size * grid_size, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return torch.tanh(x)

class ResNet(nn.Module):
    def __init__(self, grid_size: int, initial_channels: int = 1, n_residuals: int = 8):
        super().__init__()
        self.initial_block = nn.Conv2d(initial_channels, 256, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(n_residuals)])
        self.policy_head = PolicyHead(grid_size)
        self.value_head = ValueHead(grid_size)

    def forward(self, x):
        x = self.initial_block(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def main():
    resnet_test = ResNet(grid_size=5)
    test_input = torch.rand((1, 1, 5, 5))
    policy, value = resnet_test(test_input)
    print('policy:', policy)
    print('value:', value)

if __name__ == '__main__':
    main()
