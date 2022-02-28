import os
import os.path
import random
import re
from torch.utils.data import Dataset



def read_prop_ind(helper, attr='race'):
    rootdir = f"E:\dataset\lfw\\{attr}"
    print(rootdir)
    ind = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            name = re.findall(r"\d+", filename)
            # print('name', name)
            ind.append(int(name[0]))

    helper.poison_images = random.sample(ind, 50)
    print(len(ind))
    helper.poison_images_test = random.sample(ind, 150)
    for image in helper.poison_images_test:
        ind.remove(image)
    helper.poison_images_client = ind
    print(len(ind))

class myDataset(Dataset):
    def __init__(self, x_tensors, y_tensors):
        self.dataset = x_tensors
        self.targetset = y_tensors

    def __getitem__(self, index):
        self.data = self.dataset[index]
        self.label = self.targetset[index]
        return self.data, self.label

    def __len__(self):
        return len(self.dataset)