import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils_reform import *
import numpy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(23, 30):
            self.to_relu_5_3.add_module(str(x), features[x])
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        h = self.to_relu_5_3(h)
        h_relu_5_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3,h_relu_5_3)
        return out

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

class Style_loss(nn.Module):
    def __init__(self,dtype):
        super(Style_loss, self).__init__()
        self.de_re_norm = Denormalization()
        self.loss = nn.MSELoss()
        self.vgg16 = Vgg16().to(device)
        self.dtype = dtype
        self.STYLE_WEIGHT = 0.5

    def forward(self, deg_img , gen_img):
        deg_img_norm = self.de_re_norm(deg_img)
        gen_img_norm = self.de_re_norm(gen_img)
        #print(deg_img.shape,gen_img.shape)
        #get Vgg Feature

        deg_img_vgg = self.vgg16(deg_img_norm)
        gen_img_vgg = self.vgg16(gen_img_norm)

        #calculating the gram matrix
        deg_img_gram = [gram(i) for i in deg_img_vgg]
        gen_img_gram = [gram(l) for l in gen_img_vgg]

        #calculating the loss
        loss=0
        for x,y in zip(deg_img_gram, gen_img_gram):
            loss += self.loss(x,y)
        loss *= self.STYLE_WEIGHT
        return loss



class ContentLoss(nn.Module):
    def __init__(self,dtype):
        super(ContentLoss, self).__init__()
        self.criterionContent = nn.MSELoss()
        self.dtype = dtype
        self.CONTENT_WEIGHT = 10
    def forward(self , clean_img , gen_img):
        Mask = clean_img<0.0
        Mask = Mask.float()
        self.loss = self.criterionContent.forward((Mask*clean_img),(Mask*gen_img))
        return self.loss*self.CONTENT_WEIGHT



class GanLoss(nn.Module):
    def __init__(self,target_real_label = 1.0,target_fake_label =0.0,tensor=torch.FloatTensor,opt =None):
        super(GanLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.opt = opt
        self.Tensor = tensor

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def loss(self , input , target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        loss = F.binary_cross_entropy_with_logits(input, target_tensor)
        return loss




