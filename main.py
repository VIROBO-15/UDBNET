from __future__ import print_function
import argparse
import torch.utils.data
from dataloader import *
from Model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Visulaizer_reform import *
from torchvision.utils import save_image
import numpy
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/home/media/TIP_Bina/data/binarization/train_data_for_2011', help='path to dataset')
    parser.add_argument('--train_iam_list', default="/home/media/TIP_Bina/data/binarization/train_pair_for_2011.lst",
                        help='path to dataset')
    parser.add_argument('--workers', default=8, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cuda', default='cuda', help='enables cuda')
    parser.add_argument('--eval_freq_iter', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.5, help='beta2 for adam. default=0.5')
    parser.add_argument('--print_freq_iter', type=int, default=100)
    parser.add_argument('--niter_decay', type=int, default=100,help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--norm_G', type=str, default='instance',help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='instance',help='instance normalization or batch normalization')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--label_nc', type=int,  default=3, help='Number of channels in the images')


    opt = parser.parse_args()
    opt.isTrain = True
    train_loader = get_dataloader(opt)
    model = Model(opt)
    model.to(device)
    tensorboard_viz = Visualizer()

    step=0
    
    for epochs in range(15):
         for i_batch, (clean_img, degraded_img) in enumerate(train_loader):
             step = step + 1
             clean_img = clean_img.to(device)
             degraded_img = degraded_img.to(device)
             gen_img = model.forward_texture(clean_img, degraded_img)
             errors = model.get_current_errors_texture()
   
             if (step + 1) % opt.print_freq_iter == 0:
                 img = model.img_list_after_texture()
                 tensorboard_viz.vis_image(img, step)
                 print('Epoch: {}, Iter: {}, Steps: {}, Loss:{}'.format(epochs, i_batch, step, errors))
    
         torch.save(model.state_dict(), './model/checkpoint_texture_epoch_{}.pth'.format(epochs))

    #model.load_state_dict(torch.load('./model/checkpoint_texture_epoch_14.pth'))
    model.freeze_network(model.Texture_generator)
    model.freeze_network(model.Texture_Discrimator)

    steps=0
    for epoch in range(20):
        for i_batch_bin ,(clean_img,degraded_img) in enumerate(train_loader):
            steps= steps + 1
            clean_img = clean_img.to(device)
            degraded_img = degraded_img.to(device)
            gen_img = model.Texture_generator(clean_img, degraded_img)
            gen_img_bin = model.forward_binaziation(gen_img, clean_img, degraded_img)
            error_bin = model.get_current_errors_bin()
            print('Epoch: {}, Iter: {}, Steps: {}, Loss:{}'.format(epoch, i_batch_bin, step, error_bin))

            if (steps + 1) % opt.print_freq_iter == 0:
                img = model.img_list_after_bin()
                tensorboard_viz.vis_image(img, step)
                print('Epoch: {}, Iter: {}, Steps: {}, Loss:{}'.format(epoch, i_batch_bin, step, error_bin))

        torch.save(model.state_dict(), './model/checkpoint_binarization_epoch_{}.pth'.format(epoch))


    model.Un_freeze_network(model.Texture_generator)
    model.Un_freeze_network(model.Texture_Discrimator)

    steps=0
    for epoch in range(10):
        for i_batch_bin ,(clean_img,degraded_img) in enumerate(train_loader):
            steps= steps + 1
            clean_img = clean_img.to(device)
            degraded_img = degraded_img.to(device)
            model.joint_forward(clean_img, degraded_img)
            error_bin = model.get_current_errors_bin()

            if (step + 1) % opt.print_freq_iter == 0:
                img = model.img_list_after_texture()
                tensorboard_viz.vis_image(img, step)
                print('Epoch: {}, Iter: {}, Steps: {}, Loss:{}'.format(epoch, i_batch_bin, step, error_bin))

        torch.save(model.state_dict(), './model/checkpoint_joint_epoch_{}.pth'.format(epoch))







