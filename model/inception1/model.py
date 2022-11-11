import torch
import torch.nn as nn
import torchvision
from data import NoveldaDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, mode, input_channels, output_channels, alpha):
        super().__init__()
        self.conv1, self.conv3, self.conv5, self.pool = self.create_inception_block(mode, input_channels, output_channels, alpha)

    def forward(self, x):
        conv1 = self.conv1(x)
        #print("conv1", conv1.shape)
        conv3 = self.conv3(x)
        #print("conv3", conv3.shape)
        conv5 = self.conv5(x)
        #print("conv5", conv5.shape)
        pool = self.pool(x)
        #print("pool", pool.shape)
        #print("====================")
        return torch.cat([conv1, conv3, conv5, pool], 1)

    def create_single_block(self, input_channels, output_channels, conv_kernel):
        #assert conv_kernel % 2 == 1
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, conv_kernel, padding= ((conv_kernel[0]-1)//2, (conv_kernel[1]-1)//2 ) ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True))

    def create_block_with_bottleneck(self, mode, input_channels, output_channels, conv_kernel, alpha):
        if mode == 1:
            return self.create_single_block(input_channels, output_channels, conv_kernel)
        elif mode == 2:
            return nn.Sequential(
                self.create_single_block(input_channels, output_channels//alpha, 1),
                self.create_single_block(output_channels//alpha, output_channels, conv_kernel))
        elif mode == 3:
            return nn.Sequential(
                self.create_single_block(input_channels, output_channels//alpha, 1),
                self.create_single_block(output_channels//alpha, output_channels//alpha, conv_kernel),
                self.create_single_block(output_channels//alpha, output_channels, 1))

    def create_inception_block(self, mode, input_channels, output_channels, alpha):
        # mode 1 : no-bottleneck
        # mode 2 : bottleneck -> conv
        # mode 3 : bottleneck -> conv -> bottleneck
        # branches = 50%:3x3, 25%:1x1, 12.5%:5x5, 12.5%:3x3pool
        # 
        # alpha = bottleneck_ratio : conv_channel / alpha = bottleneck_channels
        assert output_channels % (8*alpha) == 0
        assert mode >= 1 and mode <= 3 and type(mode) is int
        # 1x1 conv
        conv1 = self.create_single_block(input_channels, output_channels//4, (1,1))
        # 3x3, 5x5 conv
        conv3 = self.create_block_with_bottleneck(mode, input_channels, output_channels//2, (3,5), alpha)
        conv5 = self.create_block_with_bottleneck(mode, input_channels, output_channels//8, (5,7), alpha)
        # 3x3 pool
        pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                             nn.Conv2d(input_channels, output_channels//8, 1))
        return conv1, conv3, conv5, pool

class InceptionBlock(nn.Module):
    def __init__(self, f1, f2, f3, avg_size, mode, alpha) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(avg_size)
        self.inception1 = InceptionModule(mode, f1, f2, alpha)
        self.inception2 = InceptionModule(mode, f2, f3, alpha)
        self.inception3 = InceptionModule(mode, f3, f1, alpha)

    def forward(self, x):
        x = self.pool(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        return x

class InceptionClassifier(nn.Module):
    def __init__(self, mode, alpha):
        super().__init__()
        self.inception1 = InceptionModule(mode, 1, 96, alpha)
        # pool 32 -> 16
        self.pool1 = nn.AvgPool2d(2)
        self.inception2 = InceptionModule(mode, 96, 256, alpha)
        # pool 16 ->
        self.inblock1 = InceptionBlock(256,384,256,2,mode,alpha)
        self.inblock2 = InceptionBlock(256,384,256,2,mode,alpha)
        self.inblock3 = InceptionBlock(256,384,256,4,mode,alpha)

        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.inception1(x)
        out = self.inception2(self.pool1(out))
        out = self.inblock1(out)
        out = self.inblock2(out)
        out = self.inblock3(out)
        
        out = F.avg_pool2d(out, 7).view(batch_size, -1) # Global Average Pooling
        out = self.fc(out)
        return out

if __name__ == '__main__':
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 3)
    #model = InceptionClassifier(1, alpha=2)
    model.cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    _root = "/home/vijay/Documents/devmk4/ECE599_UWB/dataset"
    dataset = NoveldaDataset(_root, "train.txt", cache=True)
    dataloader = DataLoader(dataset, batch_size=16)

    #with torch.no_grad():
    for i in range(10):
        with torch.no_grad():
            for _img, _label in tqdm(dataloader):
                _img = _img.float()
                _img = _img.cuda()
                pred = model(_img)