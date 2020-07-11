import numpy
import torch
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

class Visualizer:
    def __init__(self):
        if os.path.exists('runs'):
            shutil.rmtree('runs')

        self.writer = SummaryWriter()
        self.mean = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        self.std = torch.tensor([1 / 0.5, 1 / 0.5, 1 / 0.5]).to(device)

    def vis_image(self, visularize, step):
        for keys, value in visularize.items():
            #print(keys,value.size())

            """ Denormalization """

            value.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])

            visularize[keys] = torchvision.utils.make_grid(value)
            self.writer.add_image('{}'.format(keys), visularize[keys], step)


    def plot_current_errors(self, iters, errors):

        for keys, value in errors.items():
            #print(keys,value.size())
            self.writer.add_scalar('{}'.format(iters,keys), errors[keys])