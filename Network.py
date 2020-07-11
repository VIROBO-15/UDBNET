import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class style_encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(style_encoder, self).__init__()

        self.down1 = UNetDown(3, 32, normalize=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256, dropout=0.5)
        self.down5 = UNetDown(256, 256, dropout=0.5)
        self.down6 = UNetDown(256, 256, dropout=0.5)
        self.down7 = UNetDown(256, 256, dropout=0.5)
        self.down8 = UNetDown(256, 256, normalize=False, dropout=0.5)
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        return d8

class Texture_Generator_and_context_encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Texture_Generator_and_context_encoder, self).__init__()

        self.style_encoder = style_encoder(in_channels=3)

        self.down1 = UNetDown(in_channels, 32, normalize=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256,dropout=0.5)
        self.down5 = UNetDown(256, 256, dropout=0.5)
        self.down6 = UNetDown(256, 256, dropout=0.5)
        self.down7 = UNetDown(256, 256, dropout=0.5)
        self.down8 = UNetDown(256, 256, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)#512+256 is the dimension of the input
        self.up2 = UNetUp(768, 512, dropout=0.5)
        self.up3 = UNetUp(768, 512, dropout=0.5)
        self.up4 = UNetUp(768, 512, dropout=0.5)
        self.up5 = UNetUp(768, 128)
        self.up6 = UNetUp(256, 64)
        self.up7 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, clean, noisy):

        # U-Net generator with skip connections from encoder to decoder

        style8 = self.style_encoder.forward(noisy)

        d1 = self.down1(clean)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        con = torch.cat((d8, style8),dim=1)

        u1 = self.up1(con, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class Texture_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Texture_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels , 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128,256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A):#////////////////////make the changes
        #img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_A)


class Binarization_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Binarization_Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Binarization_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Binarization_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels , 32, normalization=False),
            *discriminator_block(32,64),
            *discriminator_block(64,128),
            *discriminator_block(128,256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A):#Here we have made the changes///////////////////////////////////////////////
        # Concatenate image and condition image by channels to produce input
        #img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_A)

class Joint_Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Joint_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2 , 32, normalization=False),
            *discriminator_block(32,64),
            *discriminator_block(64,128),
            *discriminator_block(128,256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):#Here we have made the changes///////////////////////////////////////////////
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

