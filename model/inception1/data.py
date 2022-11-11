import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cv2

random.seed(100)

def createPaths(root, keys, train_ratio=0.8):
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    for i, key in enumerate(keys):
        _folder = os.path.join(root, key)
        for _,_,f in os.walk(_folder): pass
        f = [os.path.join(root, key, _f) for _f in f]
        random.shuffle(f)
        _train_len = int(len(f)*train_ratio)
        _train_paths = f[:_train_len]
        _test_paths = f[_train_len:]
        _train_labels = [i for _ in range(len(_train_paths))]
        _test_labels = [i for _ in range(len(_test_paths))]

        train_paths += _train_paths
        test_paths += _test_paths
        train_labels += _train_labels
        test_labels += _test_labels

    print(len(train_paths), len(test_paths))
    print(len(train_labels), len(test_labels))

    with open(os.path.join(os.path.join(root, 'train.txt')), 'w') as f:
        for i, item in enumerate(train_paths):
            f.write("%s %i\n"%(item, train_labels[i]))

    with open(os.path.join(os.path.join(root, 'test.txt')), 'w') as f:
        for i, item in enumerate(test_paths):
            f.write("%s %i\n"%(item, test_labels[i]))  

class NoveldaDataset(Dataset):
    def __init__(self, root, path_file, img_size=224, cache=False) -> None:
        self.root = root
        self.keys = ["human_walking", "human_limping","human_falling"]
        self.path_file = path_file
        self.paths, self.labels = self.__load_files__()
        self.img_size = img_size
        self.shouldCache = cache
        self.cache = {}

    def preprocess(self, image):
        proc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = proc(image)
        return image

    def __load_files__(self):
        with open( os.path.join(self.root, self.path_file), 'r') as f:
            contents = list(map(str.strip, f.readlines()))
        contents = [con.split() for con in contents]
        _path = [con[0] for con in contents]
        _labels = [int(con[1]) for con in contents]
        return _path, np.array(_labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.shouldCache:
            if index in self.cache:
                return self.cache[index]

        # load in mat:
        rawData = np.loadtxt(self.paths[index])
        l = rawData.shape[0]

        I_Data = rawData[0:l//2,:]
        Q_Data = rawData[(l//2)+1:,:]
        IQ_Data = I_Data + (1j*Q_Data)
        IQ_Data = np.abs(IQ_Data)
        IQ_Data = IQ_Data.transpose()

        # clutter removal? (it works kinda)
        #IQ_Data = (IQ_Data - np.min(IQ_Data))/np.ptp(IQ_Data)
        alpha = 0.1
        clutter_bg = np.zeros_like(IQ_Data)
        clutter_bg[0,:] = IQ_Data[0,:]

        for k in range(1,IQ_Data.shape[0]):
            clutter_bg[k,:] = (alpha*clutter_bg[k-1,:])+((1-alpha)*IQ_Data[k,:])

        IQ_Data = IQ_Data - clutter_bg # looks similar to figure 13 
        
        # Histogram to remove sparse, super low values and recalib values
        test = np.histogram(IQ_Data.ravel(), bins=50)
        new_low_thres = test[1][np.argmax(test[0])]
        IQ_Data = np.clip(IQ_Data, new_low_thres, IQ_Data.max())        
        IQ_Data = IQ_Data[:,50:]
        IQ_Data = (IQ_Data - np.min(IQ_Data))/np.ptp(IQ_Data)
        IQ_Data = cv2.resize(IQ_Data, (self.img_size,self.img_size))
        IQ_Data = self.preprocess(IQ_Data)

        if self.shouldCache:
            if not index in self.cache:
                self.cache[index] = [IQ_Data, self.labels[index]]

        return IQ_Data, self.labels[index]

if __name__ == '__main__':
    _root = "/home/vijay/Documents/devmk4/ECE599_UWB/dataset"
    createPaths(_root, ["human_walking", "human_limping","human_falling"])
    dataset = NoveldaDataset(_root, "test.txt", cache=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs('data_example', exist_ok=True)

    data_check = 0
    for i in range(10):
        t1 = time.time()
        for j, (_img, _label) in enumerate(tqdm(dataloader)):
            #if data_check == _label[0]:
            if _label[0] == 1:
                plt.figure(figsize=(9,3))
                #plt.figure(figsize=(5,5))
                plt.title(["human_walking", "human_limping","human_falling"][_label[0]])
                plt.imshow(_img[0,0,:,:], cmap='jet')
                plt.ylabel("Slow Time")
                plt.xlabel("Fast Time (Range Bin)")
                plt.tight_layout()
                plt.show()
                #plt.savefig('data_example/test_%i.png'%(_label[0]))
                plt.close()
            #data_check += 2

        t2 = time.time()
        print(t2-t1)