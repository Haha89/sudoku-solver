# -*- coding: utf-8 -*-

"""Definition of the CNN and functions for training and testing"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

DEVICE = 'cpu'
N_EPOCHS = 15
LEARNING_RATE = 0.001
torch.manual_seed(42)


class Net(nn.Module):
    """Definition of the CNN """

    def __init__(self):
        """Initialisation"""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """Returns the prediction done by the network"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    network = Net().to(DEVICE)
    optimizer = optim.RMSprop(network.parameters(), lr=LEARNING_RATE,
                              alpha=0.9, eps=1e-08, weight_decay=0.0)
    lossCE = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.QMNIST('/files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ColorJitter(
                                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.QMNIST('/files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size=1000, shuffle=True)

    test_losses = []
    for epoch in range(N_EPOCHS):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = lossCE(output, target)
            loss.backward()
            optimizer.step()

        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += lossCE(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(f'Epoch {epoch+1}: Acc: {correct}/{len(test_loader.dataset)}' +
              f' ({100. * correct / len(test_loader.dataset):.2f}%)\n')

    CHECKPOINT = {'model': Net(),
                  'state_dict': network.state_dict(),
                  'optimizer' : optimizer.state_dict()}
    torch.save(CHECKPOINT, '../results/model.pth')
