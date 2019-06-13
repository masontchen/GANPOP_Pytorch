import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.spectral_norm import spectral_norm
from Basic_blocks import *
from torch.autograd import Variable

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch' or 'spectral':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    #elif norm_type == 'spectral':
        #norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
0

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], return_feature=False, hidden_channels=[32], generator='resnet_6blocks'):

    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, return_feature=return_feature)
    elif which_model_netG == 'resnet_9blocks_nogan':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, spec_norm=False)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, return_feature=return_feature)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, return_feature=return_feature)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, return_feature=return_feature)
    elif which_model_netG == 'unet_128_nogan':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, spec_norm=False)
    elif which_model_netG == 'unet_256_nogan':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             spec_norm=False)
    elif which_model_netG == 'fusion':
        netG = FusionGenerator(input_nc, output_nc, ngf, num_downs=4)
    elif which_model_netG == 'fusion_nogan':
        netG = FusionGenerator(input_nc, output_nc, ngf, num_downs=4, spec_norm=False)
    elif which_model_netG == 'lstm':
        netG = LSTMGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, hidden_channels=hidden_channels, generator=generator)
    elif which_model_netG == 'kunet':
        netG = KUnetGenerator(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', spec_norm=True, return_feature=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.return_feature = return_feature
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if spec_norm:
            model = [nn.ReflectionPad2d(3),
                     spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                               bias=use_bias)),
                     norm_layer(ngf),
                     nn.ReLU(True)]
        else:
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                                             bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if spec_norm:
                model += [spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias)),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]


        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, spec_norm=spec_norm)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if spec_norm:
                model += [spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias)),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        if self.return_feature:
            self.model = nn.Sequential(*model)
        else:
            model += [nn.ReflectionPad2d(3)]
            model += [spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))]
            model += [nn.Tanh()]
            self.model = nn.Sequential(*model)

        # model += [nn.ReflectionPad2d(3)]
        # if spec_norm:
        #     model += [spectral_norm(nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0))]
        # else:
        #     model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]
        # self.model = nn.Sequential(*model)

    def forward(self, input):
        # if self.return_feature:
        #     self.mm = nn.Sequential(*(list(self.model.children())[:-3]))
        #     return self.mm(input)
        # else:
        #     return self.model(input)

        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, spec_norm):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, spec_norm)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, spec_norm):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if spec_norm:
            conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                           norm_layer(dim),
                           nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if spec_norm:
            conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                           norm_layer(dim)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, spec_norm=True, return_feature=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, spec_norm=spec_norm)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, spec_norm=spec_norm,
                                             return_feature=return_feature)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class StackUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, spec_norm=True):
        super(StackUnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, spec_norm=spec_norm)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                                 spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, spec_norm=spec_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer, spec_norm=spec_norm)

        self.model1 = unet_block
        self.model2 = unet_block

    def forward(self, input):
        output1 = self.model1(input)
        return self.model2(output1)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, spec_norm=True, return_feature=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.return_feature = return_feature
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        # check if GAN or simple U-Net
        if spec_norm == False:
            print('spectral norm disabled')
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = norm_layer(inner_nc)
            uprelu = nn.ReLU(True)
            upnorm = norm_layer(outer_nc)

            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                conv = nn.Conv2d(inner_nc, outer_nc, kernel_size=1)
                down = [downconv]
                up = [uprelu, upconv, conv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up
        else:
            downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                               stride=2, padding=1, bias=use_bias))
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = norm_layer(inner_nc)
            uprelu = nn.ReLU(True)
            upnorm = norm_layer(outer_nc)

            if outermost:
                upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                                          kernel_size=4, stride=2,
                                                          padding=1))
                down = [downconv]
                conv = spectral_norm(nn.Conv2d(inner_nc, outer_nc, kernel_size=1))
                up = [uprelu, upconv, conv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                                          kernel_size=4, stride=2,
                                                          padding=1, bias=use_bias))
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                                          kernel_size=4, stride=2,
                                                          padding=1, bias=use_bias))
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            if self.return_feature:
                mm = nn.Sequential(*(list(self.model.children())[:-2]))
                # mm = nn.Sequential(mm, nn.Upsample(size=(256,1024)))
                return mm(x)
            else:
                return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        attention = False
        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          # kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]
        if attention:
            sequence += Self_Attn(ndf * nf_mult, 'relu')

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            # nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      # kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        if attention:
            sequence += Self_Attn(ndf * nf_mult, 'relu')

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)),
            # spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias))]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class FusionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, num_downs=4, spec_norm=True):
        super(FusionGenerator, self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")
        print "%d fusion blocks, %d filters \n" %(num_downs, self.out_dim * (2 ** num_downs))

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn, spec_norm)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn, spec_norm)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn, spec_norm)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn, spec_norm)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn, spec_norm)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2, spec_norm)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2, spec_norm)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2, spec_norm)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2, spec_norm)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2, spec_norm)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2, spec_norm)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2, spec_norm)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2, spec_norm)

        # output

        if spec_norm:
            self.out = spectral_norm(nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1))
        else:
            self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        self.out_2 = nn.Tanh()

    def forward(self, input):
        # input 400x400x3 image,
        down_1 = self.down_1(input)  # 400x400x64
        pool_1 = self.pool_1(down_1)  # 200x200x64
        down_2 = self.down_2(pool_1)  # 200x200x128
        pool_2 = self.pool_2(down_2)  # 100x100x128
        down_3 = self.down_3(pool_2)  # 100x100x256
        pool_3 = self.pool_3(down_3)  # 50x50x256
        down_4 = self.down_4(pool_3)  # 50x50x512
        pool_4 = self.pool_4(down_4)  # 25x25x512

        bridge = self.bridge(pool_4)  # 25x25x1024

        deconv_1 = self.deconv_1(bridge)  # 50x50x512
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)  # 50x50x512
        deconv_2 = self.deconv_2(up_1)  # 100x100x256
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)  # 200x200x128
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)  # 400x400x64
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)  # 400x400x3 output
        out = self.out_2(out)  # tanh
        # out = torch.clamp(out, min=-1, max=1)

        return out


class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, spec_norm=True):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn, spec_norm)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn, spec_norm)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn, spec_norm)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3




class KUnetGenerator(nn.Module):
    def __init__(self, num_channels=1, num_classes=1):
        super(KUnetGenerator, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[2], num_feat[3]))

        # self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
        #                             Conv3x3(num_feat[3], num_feat[4]))
        self.bottom = nn.Sequential(spectral_norm(nn.Conv2d(num_feat[3], num_feat[3],
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)),
                                   nn.ReLU())

        self.up1 = UpConcat(num_feat[3], num_feat[3])
        # self.upconv1 = Conv3x3(num_feat[4], num_feat[3])
        self.upconv1 = nn.Sequential(spectral_norm(nn.Conv2d(num_feat[3], num_feat[3],
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)),
                                    nn.ReLU())

        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(spectral_norm(nn.Conv2d(num_feat[0],
                                             num_classes,
                                             kernel_size=1)),
                                   nn.Tanh())

    def forward(self, inputs, return_feature=False):
        # print(inputs.data.size())
        down1_feat = self.down1(inputs)
        # print(down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print(down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print(down3_feat.size())
        down4_feat = self.down4(down3_feat)
        # print(down4_feat.size())
        bottom_feat = self.bottom(down4_feat)

        # print(bottom_feat.size())
        up1_feat = self.up1(bottom_feat, down4_feat)
        # print(up1_feat.size())
        up1_feat = self.upconv1(up1_feat)
        # print(up1_feat.size())
        up2_feat = self.up2(up1_feat, down3_feat)
        # print(up2_feat.size())
        up2_feat = self.upconv2(up2_feat)
        # print(up2_feat.size())
        up3_feat = self.up3(up2_feat, down2_feat)
        # print(up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print(up3_feat.size())
        up4_feat = self.up4(up3_feat, down1_feat)
        # print(up4_feat.size())
        up4_feat = self.upconv4(up4_feat)
        # print(up4_feat.size())

        if return_feature:
            outputs = up4_feat
        else:
            outputs = self.final(up4_feat)

        return outputs



class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Drop(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Drop, self).__init__()

        self.conv1 = nn.Sequential(spectral_norm(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(spectral_norm(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = spectral_norm(nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2))

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        # print(out.size())
        return out


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv = spectral_norm(nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2))

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        # outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out


class LSTMGenerator(nn.Module):
    def __init__(self, input_nc=64, output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 hidden_channels=[32], generator='resnet_6blocks'):
        super(LSTMGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        if generator == 'resnet_6blocks':
            self.model1 = ResnetGenerator(output_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks=6,
                                   return_feature=True)

        model = BDCLSTM(input_channels=self.input_nc, hidden_channels = hidden_channels, num_classes=output_nc, kernel_size=3)
        self.model = model


    def forward(self, A1, A, A3):
        x = torch.cat((A1, A, A3), dim=0)
        y1 = self.model1(x)
        x1 = y1[0, :, :, :]
        x = y1[1, :, :, :]
        x3 = y1[2, :, :, :]
        x1 = torch.unsqueeze(x1, dim=0)
        x = torch.unsqueeze(x, dim=0)
        x3 = torch.unsqueeze(x3, dim=0)
        # x1 = self.model1(A1)
        # x = self.model1(A)
        # x3 = self.model1(A3)

        return self.model(x1, x, x3)

        # return self.model(A1, A, A3)


class CLSTMCell(nn.Module):

    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(CLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(self.input_channels + self.hidden_channels,
                              self.num_features * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)),
                              nn.Dropout(0.1))


    # Forward propogation formulation
    def forward(self, x, h, c):
        # print('x: ', x.type)
        # print('h: ', h.type)
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)

        # NOTE: A? = xz * Wx? + hz-1 * Wh? + b? where * is convolution
        (Ai, Af, Ao, Ag) = torch.split(A,
                                       A.size()[1] // self.num_features,
                                       dim=1)

        i = torch.tanh(Ai)     # input gate
        f = torch.tanh(Af)     # forget gate
        o = torch.tanh(Ao)     # output gate
        g = torch.tanh(Ag)

        c = c * f + i * g           # cell activation state
        h = o * torch.tanh(c)     # cell hidden state

        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda(),
               Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda())
        except:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])),
                    Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])))


''' Class CLSTM.
    This represents a series of CLSTM nodes (one direction)
'''


class CLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[32],
                 kernel_size=5, bias=True):
        super(CLSTM, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                # populate hidden states for all layers
                if step == 0:
                    (h, c) = CLSTMCell.init_hidden(bsize,
                                                   self.hidden_channels[layer],
                                                   (height, width))
                    internal_state.append((h, c))

                # do forward
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]

                input, c = getattr(self, name)(
                    input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)

            outputs.append(input)

        #for i in range(len(outputs)):
        #    print(outputs[i].size())
        return outputs


class BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[32],
                 kernel_size=5, bias=True, num_classes=1):

        super(BDCLSTM, self).__init__()
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.reverse_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.pad = nn.ReflectionPad2d(3)
        self.conv = spectral_norm(nn.Conv2d(
            2 * hidden_channels[-1], num_classes, kernel_size=7))
        # self.soft = nn.Softmax2d()
        self.soft = nn.Tanh()

    # Forward propogation
    # x --> BatchSize x NumChannels x Height x Width
    #       BatchSize x 64 x 240 x 240
    def forward(self, x1, x2, x3):
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x3 = torch.unsqueeze(x3, dim=1)

        xforward = torch.cat((x1, x2), dim=1)
        xreverse = torch.cat((x3, x2), dim=1)

        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)

        # assumes y is BatchSize x NumClasses x 240 x 240
        # print(yforward[-1].type)
        ycat = torch.cat((yforward[-1], yreverse[-1]), dim=1)

        # print(ycat.size())
        y = self.pad(ycat)
        y = self.conv(y)
        # print(y.type)
        y = self.soft(y)
        # print(y.size())
        return y
