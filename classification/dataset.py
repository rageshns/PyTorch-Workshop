from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
from PIL import Image

class Custom_Dataset(Dataset):
    def __init__(self):
        transform = transforms.ToTensor()
        self.dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)

    def __getitem__(self, idx):
        image = self.dataset[self.sorted_indices[idx]][0]
        label = self.dataset[self.sorted_indices[idx]][1]
        return image, label
    
    def __len__(self):
        return len(self.dataset)

class Custom_Dataset_Balanced(Dataset):
    def __init__(self):
        transform = transforms.ToTensor()
        self.dst_train = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        x = np.array(self.dst_train.targets)
        sorted_indices = np.argsort(x)
        num_elem_class = int(np.sum(x==0))
        s = np.zeros((10,num_elem_class),int)
        for i in range(10):
            s[i] = sorted_indices[i*num_elem_class:(i+1)*num_elem_class]

        self.sorted_indices = [item for indices in zip(s[0], s[1], s[2],s[3],s[4], s[5],s[6],s[7],s[8],s[9]) for item in indices]

    def __getitem__(self, idx):
        image = self.dst_train[self.sorted_indices[idx]][0]
        label = self.dst_train[self.sorted_indices[idx]][1]
        return image, label
    
    def __len__(self):
        return len(self.dst_train)
    
class Custom_Dataset_Balanced_Split(Dataset):
    def __init__(self,split="train"):
        if split == "train" or split == "val":
            is_train = True
            
        transform = transforms.ToTensor()
        self.dst_train = datasets.CIFAR10('data', train=is_train, download=True, transform=transform)
        
        x = np.array(self.dst_train.targets)
        sorted_indices = np.argsort(x)
        num_elem_class = int(np.sum(x==0))
        s = np.zeros((10,num_elem_class),int)
        for i in range(10):
            s[i] = sorted_indices[i*num_elem_class:(i+1)*num_elem_class]

        self.sorted_indices = [item for indices in zip(s[0], s[1], s[2],s[3],s[4], s[5],s[6],s[7],s[8],s[9]) for item in indices]
        
        # lets use 20% of the trainning dataset to validation
        if split == "train":
            self.sorted_indices = self.sorted_indices[0:int(len(self.sorted_indices)*0.8)]
        elif split == "val":
            self.sorted_indices = self.sorted_indices[int(len(self.sorted_indices)*0.8):]

    def __getitem__(self, idx):
        image = self.dst_train[self.sorted_indices[idx]][0]
        label = self.dst_train[self.sorted_indices[idx]][1]
        return image, label
    
    def __len__(self):
        return len(self.sorted_indices)
    
from torch.utils.data import Dataset,DataLoader
batch_size = 4   
train_set = Custom_Dataset(split="train")
val_set = Custom_Dataset(split="val")
test_set =  Custom_Dataset(split="test")

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)



for i, data in enumerate(trainloader):
    inputs, labels = data
    print(labels)
