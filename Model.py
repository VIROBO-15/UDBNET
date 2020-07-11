from Network_reform import *
import torch
import torch.nn as nn
from collections import OrderedDict
from loss_reform import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()
        self.opt = opt
        self.Texture_generator = Texture_Generator_and_context_encoder()
        self.Texture_Discrimator = Texture_Discriminator()
        self.Binarization_generator = Binarization_Generator()
        self.Binarization_Discrimator = Binarization_Discriminator()
        self.joint_discriminator = Joint_Discriminator()

        self.FloatTensor = torch.cuda.FloatTensor

        self.Texture_generator, self.Texture_Discrimator, self.Binarization_generator, self.Binarization_Discrimator, self.joint_discriminator = self.initialize_networks(opt)

        self.style_loss = Style_loss(self.FloatTensor)
        self.ContentLoss = ContentLoss(self.FloatTensor)
        self.GanLoss= GanLoss(tensor=self.FloatTensor)
        self.criterionFeat = torch.nn.L1Loss()
        self.optimizer_G_texture, self.optimizer_D_texture,self.optimizer_G_bin, self.optimizer_D_bin, self.optimizer_joint = self.create_optimizers(opt)


    def initialize_networks(self, opt):
        self.Texture_generator.apply(weights_init_normal).to(device)
        self.Texture_Discrimator.apply(weights_init_normal).to(device)

        self.Binarization_generator.apply(weights_init_normal).to(device)
        self.Binarization_Discrimator.apply(weights_init_normal).to(device)

        self.joint_discriminator.apply(weights_init_normal).to(device)

        return self.Texture_generator,self.Texture_Discrimator, self.Binarization_generator, self.Binarization_Discrimator, self.joint_discriminator


    def create_optimizers(self, opt):
        G_params_texture = list(self.Texture_generator.parameters())
        G_params_binarization = list(self.Binarization_generator.parameters())
        if opt.isTrain:
            D_params_texture = list(self.Texture_Discrimator.parameters())
            D_params_binarization = list(self.Binarization_Discrimator.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = opt.lr, opt.lr
        optimizer_G_texture = torch.optim.Adam(G_params_texture, lr=G_lr, betas=(beta1, beta2))
        optimizer_D_texture = torch.optim.Adam(D_params_texture, lr=D_lr, betas=(beta1, beta2))

        optimizer_G_binarization = torch.optim.Adam(G_params_binarization, lr=G_lr, betas=(beta1, beta2))
        optimizer_D_binarization = torch.optim.Adam(D_params_binarization, lr=D_lr, betas=(beta1, beta2))

        optimizer_D_joint = torch.optim.Adam(D_params_binarization, lr=D_lr, betas=(beta1, beta2))



        return optimizer_G_texture, optimizer_D_texture, optimizer_G_binarization, optimizer_D_binarization, optimizer_D_joint


    def forward_texture(self,clean_img , deg_img):

        self.optimizer_G_texture.zero_grad()

        self.G_losses_texture = {}
        self.clean_img = clean_img
        self.deg_img = deg_img

        self.gen_img = self.Texture_generator.forward(self.clean_img, self.deg_img)
        D_fake = self.Texture_Discrimator(self.gen_img)

        self.G_losses_texture['GAN'] = self.GanLoss.loss(D_fake, True)
        self.G_losses_texture['style_loss'] = self.style_loss(self.deg_img, self.gen_img)
        self.G_losses_texture['Content_Loss'] = self.ContentLoss(self.clean_img, self.gen_img)


        G_losses_texture_ = sum(self.G_losses_texture.values())
        G_losses_texture_.backward()
        self.optimizer_G_texture.step()


        self.D_losses_texture = {}

        self.optimizer_D_texture.zero_grad()
        self.gen_img = self.gen_img.detach()

        D_fake = self.Texture_Discrimator(self.gen_img)
        D_real_Ir = self.Texture_Discrimator(self.deg_img)

        self.D_losses_texture['D_fake'] = self.GanLoss.loss(D_fake, False)
        self.D_losses_texture['D_real'] = self.GanLoss.loss(D_real_Ir, True)


        D_losses_texture_ = sum(self.D_losses_texture.values()).mean()
        D_losses_texture_.backward()
        self.optimizer_D_texture.step()

        return self.gen_img

    def forward_binaziation(self, gen_img, clean_img, degraded_img):
        self.optimizer_G_bin.zero_grad()
        G_losses_bin ={}
        self.Bin_clean_img = self.Binarization_generator(gen_img)
        self.clean_img = clean_img
        self.deg_img = degraded_img
        self.gen_img = gen_img

        D_bin_fake = self.Binarization_Discrimator(self.Bin_clean_img)
        G_losses_bin['GAN'] = self.GanLoss.loss(D_bin_fake,True)
        G_losses_bin['pixel_loss'] = self.criterionFeat(clean_img, self.Bin_clean_img)
        # G_losses_bin['pixel_loss'] = F.mse_loss(clean_img, self.Bin_clean_img, reduction='sum') / gen_img.shape[0]

        self.G_losses_bin = G_losses_bin
        # G_losses_bin_ = sum(G_losses_bin.values()).mean()
        G_losses_bin_ = G_losses_bin['GAN'] + 100*G_losses_bin['pixel_loss']
        G_losses_bin_.backward()
        self.optimizer_G_bin.step()

        D_losses_bin ={}
        self.optimizer_D_bin.zero_grad()
        self.Bin_clean_img =self.Bin_clean_img.detach()
        D_fake = self.Binarization_Discrimator(self.Bin_clean_img)
        D_real_Ir=self.Binarization_Discrimator(clean_img)
        D_losses_bin['D_fake'] = self.GanLoss.loss(D_fake,False)
        D_losses_bin['D_real'] = self.GanLoss.loss(D_real_Ir, True)
        self.D_losses_bin = D_losses_bin
        D_losses_bin_ = sum(D_losses_bin.values()).mean()
        D_losses_bin_.backward()
        self.optimizer_D_bin.step()
        return self.Bin_clean_img

    def joint_forward(self,clean_img, deg_img):

        self.optimizer_joint.zero_grad()#now added

        #--------------------------Tecture Network______________________#

        self.optimizer_G_texture.zero_grad()

        self.G_losses_texture = {}
        self.clean_img = clean_img
        self.deg_img = deg_img

        self.gen_img = self.Texture_generator.forward(self.clean_img, self.deg_img)
        D_fake = self.Texture_Discrimator(self.gen_img)

        D_pair_1 = self.joint_discriminator(self.gen_img, clean_img)#--------------------now added-------#

        self.G_losses_texture['GAN'] = self.GanLoss.loss(D_fake, True)
        self.G_losses_texture['style_loss'] = self.style_loss(self.deg_img, self.gen_img)
        self.G_losses_texture['Content_Loss'] = self.ContentLoss(self.clean_img, self.gen_img)
        self.G_losses_texture['crit_joint_texture'] = self.GanLoss.loss(D_pair_1, True)#--------------------now added-------#


        G_losses_texture_ = sum(self.G_losses_texture.values())
        G_losses_texture_.backward(retain_graph = True)
        self.optimizer_G_texture.step()


        self.D_losses_texture = {}

        self.optimizer_D_texture.zero_grad()

        D_fake = self.Texture_Discrimator(self.gen_img.detach())
        D_real_Ir = self.Texture_Discrimator(self.deg_img)

        self.D_losses_texture['D_fake'] = self.GanLoss.loss(D_fake, False)
        self.D_losses_texture['D_real'] = self.GanLoss.loss(D_real_Ir, True)


        D_losses_texture_ = sum(self.D_losses_texture.values()).mean()
        D_losses_texture_.backward(retain_graph = True)
        self.optimizer_D_texture.step()

        #-------------------------------------------------------------------------------------#


        #--------------------------------Binarization Network------------------------------------#

        self.optimizer_G_bin.zero_grad()
        G_losses_bin ={}

        self.Bin_clean_img = self.Binarization_generator(self.gen_img)
        self.Bin_clean_img_joint = self.Binarization_generator(deg_img)

        D_bin_fake = self.Binarization_Discrimator(self.Bin_clean_img)

        D_pair_2 = self.joint_discriminator(self.Bin_clean_img_joint, deg_img)#--------------------now added-------#

        G_losses_bin['GAN'] = self.GanLoss.loss(D_bin_fake,True)
        G_losses_bin['pixel_loss'] = self.criterionFeat(clean_img, self.Bin_clean_img)
        self.G_losses_texture['crit_joint_binariztion'] = self.GanLoss.loss(D_pair_2, False)#--------------------now added-------#

        self.G_losses_bin = G_losses_bin
        G_losses_bin_ = sum(G_losses_bin.values()).mean()
        G_losses_bin_.backward(retain_graph=True)
        self.optimizer_G_bin.step()

        D_losses_bin ={}
        self.optimizer_D_bin.zero_grad()
        self.Bin_clean_img =self.Bin_clean_img.detach()
        D_fake = self.Binarization_Discrimator(self.Bin_clean_img)
        D_real_Ir=self.Binarization_Discrimator(clean_img)
        D_losses_bin['D_fake'] = self.GanLoss.loss(D_fake,False)
        D_losses_bin['D_real'] = self.GanLoss.loss(D_real_Ir, True)
        self.D_losses_bin = D_losses_bin
        D_losses_bin_ = sum(D_losses_bin.values()).mean()
        D_losses_bin_.backward(retain_graph=True)
        self.optimizer_D_bin.step()

        #----------------------------------------------------------------------------#

        self.D_joint ={}
        self.D_joint['crit_joint_texture'] = self.GanLoss.loss(D_pair_1, False)
        self.G_losses_texture['crit_joint_binariztion'] = self.GanLoss.loss(D_pair_2, True)
        D_losses_joint = sum(self.D_joint.values()).mean()
        D_losses_joint.backward(retain_graph=True)

        self.optimizer_joint.step()#--------------------now added-------#


    def get_current_errors_texture(self):
        curr_losses_g_texture = {}
        curr_losses_d_texture = {}
        for k,v in self.G_losses_texture.items():
            curr_losses_g_texture[k] = v.item()
        for k, v in self.D_losses_texture.items():
            curr_losses_d_texture[k] = v.item()
        return OrderedDict({**curr_losses_g_texture, **curr_losses_d_texture})

    def get_current_errors_bin(self):
        curr_losses_g_bin = {}
        curr_losses_d_bin = {}
        for k,v in self.G_losses_bin.items():
            curr_losses_g_bin[k] = v.item()
        for k, v in self.D_losses_bin.items():
            curr_losses_d_bin[k] = v.item()
        return OrderedDict({**curr_losses_g_bin, **curr_losses_d_bin})


    def img_list_after_texture(self):
        visual_ret = OrderedDict()
        visual_ret['clean_img']= self.clean_img
        visual_ret['Degraded_img'] = self.deg_img
        visual_ret['Clean_texture_img'] = self.gen_img
        return visual_ret


    def img_list_after_bin(self):
        visual_ret = OrderedDict()
        visual_ret['clean_img']= self.clean_img
        visual_ret['Degraded_img'] = self.deg_img
        visual_ret['Clean_texture_img'] = self.gen_img
        visual_ret['Clean_bin_img'] = self.Bin_clean_img
        return visual_ret

    def freeze_network(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def Un_freeze_network(self, model):
        for param in model.parameters():
            param.requires_grad = True








