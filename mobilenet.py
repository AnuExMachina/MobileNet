import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import PIL
import torchvision
import pytorch_lightning as pl


class MobileNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = os.listdir(root_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = PIL.Image.open(os.path.join(self.root_dir, self.imgs[idx]))
        image = self.transform(image)
        if self.imgs[idx].split('.')[0] == 'dog':
            isPies = torch.tensor([1], dtype=torch.float32)
        elif self.imgs[idx].split('.')[0] == 'cat':
            isPies = torch.tensor([0], dtype=torch.float32)

        return image, isPies


class InvertedResidual(nn.Module):
    def __init__(self, inp_channels, expand_ratio):
        super().__init__()
        hidden_dim = round(inp_channels * expand_ratio)
        self.conv1 = nn.Conv2d(in_channels=inp_channels, out_channels=hidden_dim, kernel_size=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=inp_channels, kernel_size=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(inp_channels)

    def forward(self, inp):
        x = self.conv1(inp)
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        x = self.conv3(x)
        x = self.norm3(x)
        x += inp
        return x        



class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.block11 = InvertedResidual(32, 6)
        self.block12 = InvertedResidual(32, 6)
        self.block13 = InvertedResidual(32, 6)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.block21 = InvertedResidual(64, 6)
        self.block22 = InvertedResidual(64, 6)
        self.block23 = InvertedResidual(64, 6)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.block31 = InvertedResidual(128, 6)
        self.block32 = InvertedResidual(128, 6)
        self.block33 = InvertedResidual(128, 6)
        self.linear1 = nn.Linear(in_features=128, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.conv3(x)
        x = F.relu(self.norm3(x))
        x = self.block31(x)
        x = self.block32(x)
        x = self.block33(x)
        x = torch.mean(x, dim=(2, 3))
        x = F.relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log('train_accuracy', pl.metrics.functional.accuracy(torch.round(y_pred), y.int()), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log('val_accuracy', pl.metrics.functional.accuracy(torch.round(y_pred), y.int()), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log('test_accuracy', pl.metrics.functional.accuracy(torch.round(y_pred), y.int()), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    path = 'dogs-vs-cats/train/train'
    train_dataset, test_dataset = torch.utils.data.random_split(MobileNetDataset(path, transform), [20000, 5000])
    model = NeuralNetwork()

    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(model)
