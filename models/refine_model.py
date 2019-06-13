import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools


class RefineModel(BaseModel):
    def name(self):
        return 'RefineModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, gan_mode = 'vanilla', norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L11', type=float, default=100.0, help='weight for L1 loss for G1')
            parser.add_argument('--lambda_L12', type=float, default=100.0, help='weight for L1 loss for G2')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight for GAN loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN1', 'G_GAN2', 'G_L11', 'G_L12', 'D_real1', 'D_fake1', 'D_real2', 'D_fake2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B1', 'fake_B2', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D1', 'D2']
        else:  # during test time, only load Gs
            self.model_names = ['G1', 'G2']
        # load/define networks
        self.netG1 = networks0.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks0.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain,
                                        self.gpu_ids)

        if self.isTrain:
            use_sigmoid = False
            self.netD1 = networks0.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks0.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                            self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks0.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B1 = self.netG1(self.real_A)
        self.fake_B2 = self.netG2(self.fake_B1)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB1 = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B1), 1))
        pred_fake1 = self.netD1(fake_AB1.detach())
        self.loss_D_fake1 = self.criterionGAN(pred_fake1, False)
        fake_AB2 = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B2), 1))
        pred_fake2 = self.netD2(fake_AB2.detach())
        self.loss_D_fake2 = self.criterionGAN(pred_fake2, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real1 = self.netD1(real_AB)
        self.loss_D_real1 = self.criterionGAN(pred_real1, True)
        pred_real2 = self.netD2(real_AB)
        self.loss_D_real2 = self.criterionGAN(pred_real2, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake1 + self.loss_D_real1) * 0.5 + (self.loss_D_fake2 + self.loss_D_real2) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB1 = torch.cat((self.real_A, self.fake_B1), 1)
        pred_fake1 = self.netD1(fake_AB1)
        self.loss_G_GAN1 = self.criterionGAN(pred_fake1, True)
        fake_AB2 = torch.cat((self.real_A, self.fake_B2), 1)
        pred_fake2 = self.netD2(fake_AB2)
        self.loss_G_GAN2 = self.criterionGAN(pred_fake2, True)

        # Second, G(A) = B

        self.loss_G_L11 = self.criterionL1(self.fake_B1, self.real_B)
        self.loss_G_L12 = self.criterionL1(self.fake_B2, self.real_B)

        self.loss_G = self.loss_G_GAN1 * self.opt.lambda_GAN + self.loss_G_L11 * self.opt.lambda_L11 + self.loss_G_GAN2 * self.opt.lambda_GAN + self.loss_G_L12 * self.opt.lambda_L12

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad([self.netD1, self.netD2], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD1, self.netD2], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

