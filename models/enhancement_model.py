import logging
from collections import OrderedDict
import open_clip
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.archs.onestage import *
from models.base_model import BaseModel

from models.loss import CharbonnierLoss, VGGLoss, SSIM, StyleLoss, TVLoss, WGANLoss, CombinedFourierLoss, CannyLoss

logger = logging.getLogger('base')

class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        daclip_path = opt['clip_path']
        self.faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.segmentation_model.eval()
        self.daclip, self.preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=daclip_path,
                                                                              device=self.device)

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum').to(self.device)

            self.cri_vgg = VGGLoss()
            self.ssim_loss = SSIM()
            self.style_loss = StyleLoss()
            self.TV_loss = TVLoss()
            self.l1_loss = torch.nn.L1Loss()
            self.gan_loss = WGANLoss('wgan-gp')
            self.combined_fourier_loss = CombinedFourierLoss(amplitude_weight=1.0, phase_weight=1.0, loss_type='l1')
            self.adversarial_loss = nn.BCELoss()
            self.canny_loss = CannyLoss()
            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()


            self.log_dict = OrderedDict()

    def combine_elements(self, arr1, arr2):
        combine_dict = {}
        for i in range(len(arr1)):
            combine_dict[i] = {arr1[i], arr2[i]}
        combine_list = list(combine_dict.values())
        return combine_list

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)
        self.fr = data['FR'].to(self.device)
        self.L_YCrbr = data['LQs_YCrBr'].to(self.device)[:, 0:1, :, :]
        self.H_YCrbr = data['GT_YCrBr'].to(self.device)[:, 0:1, :, :]
        self.Y_Map = torch.abs(self.H_YCrbr - self.L_YCrbr) / (self.H_YCrbr + 0.00001)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        # self.dataLoader = self.combine_elements(data['GT'], data['LQS'])

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H, self.refine_H  = self.netG(self.var_L, self.daclip, self.segmentation_model)
        _, _, H, W = self.real_H.shape
        l_pix1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)*0.2
        l_pix = self.l_pix_w * self.cri_pix(self.refine_H, self.real_H)
        l_ssim = (1 - self.ssim_loss(self.refine_H, self.real_H)) * 0.5
        l_canny = self.canny_loss(self.refine_H, self.real_H) * 0.1

        l_final = l_pix +  l_ssim + l_canny + l_pix1
        l_final.backward()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.01)
        self.optimizer_G.step()
        self.log_dict['l_pix1'] = l_pix1.item()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_ssim'] = l_ssim.item()
        self.log_dict['l_canny'] = l_canny.item()


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, self.refine_H = self.netG(self.var_L, self.daclip, self.segmentation_model)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.refine_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.real_H
        del self.var_L
        del self.refine_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
