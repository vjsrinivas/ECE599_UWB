import os
import sys
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from data import NoveldaDataset
from model import InceptionClassifier
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.models import resnet18

def test_acc(model, loader, writer, epoch, writer_label_1='val/acc', writer_label_2='val/fig'):
    confusion_matrix = torch.zeros(3, 3)
    with torch.no_grad():
        for i, (dcts, annos) in enumerate(tqdm(loader)):
            dcts = dcts.float()
            dcts = dcts.cuda()
            annos = annos.cuda()
            preds = model(dcts)
            #preds = preds[:,:,0,0]
            preds = nn.functional.softmax(preds, dim=1)
            pred, class_idx = torch.max(preds, dim=1)
            for t, p in zip(annos.view(-1), class_idx.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        avg_class_acc = torch.mean(per_class_acc)
        writer.add_scalar(writer_label_1, avg_class_acc, epoch)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix.numpy())
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        disp.plot(include_values=False, ax=ax)
        fig.tight_layout()
        writer.add_figure(writer_label_2, fig, epoch, close=True)
        return avg_class_acc

def generate_folder(root, template="ved"):
    os.makedirs(root, exist_ok=True)
    i = 0
    while True:
        _cur_path = os.path.join(root, "%s_%i"%(template,i))
        if os.path.exists( _cur_path ):
            i += 1
        else:
            break
    return _cur_path

if __name__ == '__main__':
    folder = generate_folder('runs_fcn')
    writer = SummaryWriter(folder)
    #model = InceptionClassifier(1, alpha=2)
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 3)

    model.cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    loss = nn.CrossEntropyLoss()
    #opt = torch.optim.Adam(model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
    EPOCH = 500
    BATCH_SIZE = 16
    loss.cuda()
    best_loss = sys.maxsize
    best_acc = 0.0

    _root = "/home/vijay/Documents/devmk4/ECE599_UWB/dataset"
    train_dataset = NoveldaDataset(_root, "train.txt", cache=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = NoveldaDataset(_root, "test.txt", cache=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #with torch.no_grad():
    for epoch in range(EPOCH):
        avg_entropy_loss, total_items = 0.0, 0
        for i, (_img, _label) in enumerate(tqdm(train_dataloader)):
            _img = _img.float()
            _img = _img.cuda()
            _label = _label.cuda()
            preds = model(_img)
            _loss = loss(preds, _label.long())
            avg_entropy_loss += _loss.item()
            total_items += 1
            _loss.backward()
            opt.step()

        avg_entropy_loss /= total_items

        print("Average loss: ", avg_entropy_loss)
        writer.add_scalar('train/entropy_loss', avg_entropy_loss, epoch)
        
        if epoch % 5 == 0:
            model.eval()
            test_acc(model, train_dataloader, writer, epoch, writer_label_1='train/acc', writer_label_2='train/fig')
            model.train()

            model.eval()
            avg_acc = test_acc(model, test_dataloader, writer, epoch)
            if avg_acc < best_acc:
                best_loss = avg_acc
                torch.save(model.state_dict(), os.path.join(folder, 'best_acc.pth'))
            model.train()
    
    writer.close()
    torch.save(model.state_dict(), os.path.join(folder, 'last.pth'))