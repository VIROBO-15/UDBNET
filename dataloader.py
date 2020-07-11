import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import numpy

class binarization_dataset(Dataset):
    def __init__(self, root_dir, un_paired=True):

        self.root_dir = root_dir

        with open('Train_List.pickle' ,'rb') as handle:
            self.Train_Noisy_List, self.Train_Clean_List = pickle.load(handle)

        if un_paired:
            random.shuffle(self.Train_Noisy_List)
            random.shuffle(self.Train_Clean_List)

        transform_list_rgb = [transforms.Resize((256, 256)), transforms.ToTensor(),]
        self.transform_normalze = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        self.transform_doc_rgb = transforms.Compose(transform_list_rgb)


    def __len__(self):
            return len(self.Train_Noisy_List)


    def __getitem__(self, item):

        deg_img_rgb = Image.open(os.path.join(self.root_dir, self.Train_Noisy_List[item])).convert('RGB')
        clean_img_rgb = Image.open(os.path.join(self.root_dir, self.Train_Clean_List[item])).convert('RGB')

        clean_img_rgb = self.transform_doc_rgb(clean_img_rgb) # noise image
        deg_img_rgb = self.transform_doc_rgb(deg_img_rgb) # clean image

        clean_img_rgb = 1. - clean_img_rgb
        clean_img_rgb = self.transform_normalze(clean_img_rgb)
        deg_img_rgb = self.transform_normalze(deg_img_rgb)

        return clean_img_rgb, deg_img_rgb


def get_dataloader(opt):

    trainset = binarization_dataset(root_dir = opt.root_dir)
    dataloader_train = DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=False)

    return dataloader_train