import sys

sys.path.append('/home/flag54/Documents/dataSetAugument')
import xmlSet
import jaccard

import torch
import torchvision
import torch.utils.data as data


class my_loader(data.Dataset):
    def __init__(self, root, label,
                 transform=None, target_transform=None):
        super(my_loader,self).__init__()
        self.img = xmlSet.test()  # data,label

    def __getitem__(self, item):
        return self.img

    def getName(self):
        _, label = self.img
        key_img = label.keys()
        return list(key_img)

def load_one_test():
    my_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    imgLoader = torch.utils.data.DataLoader(
        my_loader(root="root", label="label_path", transform=my_transform),
        batch_size=1,shuffle=False
    )
    print 'o'
    return imgLoader

if __name__ == "__main__":
    load_one_test()