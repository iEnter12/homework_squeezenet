import argparse
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from SqueezeNet import MySqueezeNet
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--m', type=float, default=0.09, help='momentum')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--epoch', type=int, default=10, help='train epoch')
parser.add_argument('--version', type=int, default=1, help='squeezenet version(1/2)')

opt = parser.parse_args()

# 数据集转换参数
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop(224),
    transforms.Resize(224),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载训练集与测试集
train_data = datasets.CIFAR10(
    root='./CIFAR10/',
    train=True,         # 是 train 集
    download=True,      # 如果该路径没有该数据集，就下载
    transform=transform # 数据集转换参数
)


train_loader = DataLoader(train_data, shuffle=True, batch_size=opt.batch_size)


class Trainer:
    def __init__(self, opt, model, train_loader):
        self.opt = opt
        self.model = model
        self.train_loader = train_loader
        self.losses = []

    def train(self):
        loss_fn = nn.CrossEntropyLoss()
        learning_rate = opt.lr
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=opt.m,
            weight_decay=opt.wd
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        for epoch in tqdm(range(opt.epoch)):
            scheduler.step()
            for i, (X, Y) in enumerate(self.train_loader):
                X, Y = X.to('cuda:0'), Y.to('cuda:0')
                Pred = self.model(X)
                loss = loss_fn(Pred, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % opt.epoch == 0:
                    self.losses.append(loss.item())
                    print('Epoch [{}/{}], Batch [{}/{}], 训练误差: {:.4f}'.format(epoch+1, self.opt.epoch, i+1, len(self.train_loader), loss.item()))
            
            self.printParamters()
            self.saveParameters(root='./model_checkpoints/')
        
        self.plotLoss()

    def printParamters(self):
        # 打印模型的 state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def saveParameters(self, root='./'):
        try:
            os.makedirs(root)
        except OSError:
            pass
        # state_dict()表示只保存训练好的权重
        torch.save(self.model.state_dict(), root + 'squeeze_model_' + 'version' + str(opt.version) + '_epoch' + str(opt.epoch) + '.pt')

    def plotLoss(self):
        Fig = plt.figure()
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()



model = MySqueezeNet(version=opt.version, num_classes=10).to('cuda:0')
trainer = Trainer(opt, model, train_loader)
trainer.train()