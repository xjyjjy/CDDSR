import argparse
import os
import sys

import yaml
import cProfile
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
# from datasets.data import DataLoader
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import GradualWarmupScheduler
from torchsummary import summary

import torch.nn.functional as F

from torch.utils.data.sampler import Sampler
import datasets
import model as models
# import pytorch_msssim, pytorch_ssim
import utils
import random
import numpy as np
from test_blind_ours_5_2_7__5_f3_d_c15_cdcs_1 import eval_psnr
from collections import OrderedDict
from utils import channel_distribute

# from blind_model_2_1.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# #from blind_model_2_1.predictor import SamPredictor
# from blind_model_2_1.segment_anything.build_sam import build_sam_vit_b


def load_network(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net = load_net['model']['sd']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            if k in network.state_dict():
                if v.shape == network.state_dict()[k].shape:
                    load_net_clean[k] = v
                else:
                    print(k)
                    print("no")
            else:
                print(k)
                print("no")
    network.load_state_dict(load_net_clean, strict=strict)
    return network

def load_network_exSFT(load_path, network, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    load_net = torch.load(load_path)
    load_net = load_net['model']['sd']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    return network

    
def load_G(load_path_G, netG, strict_load=True):
    if load_path_G is not None:
        log('Loading model for G [{:s}] ...'.format(load_path_G))
        load_network(load_path_G, netG, strict_load)
def load_K(load_path_K, netG, strict_load=True):
    if load_path_K is not None:
        log('Loading model for K [{:s}] ...'.format(load_path_K))
        load_network(load_path_K, netG, strict_load)
def load_G_except_SFT(load_path_K, netG, strict_load=True):
    if load_path_K is not None:
        log('Loading model for K [{:s}] ...'.format(load_path_K))
        load_network_exSFT(load_path_K, netG, strict_load)

class PCAEncoder(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, l, l]
        a = self.weight.expand((B,) + self.size)
        return torch.bmm(
            batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)
        ).view((B, -1))
class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        ## label shape batch_size x 1 x h x w
          #label = label[:, 0, :, :].long()
        label = label.long()
        loss = self.criterion(self.softmax(out), label)

        return loss
    
def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    array = []
    with np.load(spec['root_path_KernelWarehouse']) as npzfile:
        array = npzfile['arr_0']
    kernel_warehouse = torch.from_numpy(array)
    pca_matrix = torch.load(spec['pca_matrix_path'])[None]
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'KernelWarehouse': kernel_warehouse, 
                                                   'pca_matrix' :pca_matrix})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    # for k, v in dataset[0].items():
    #      log('  {}: shape={} max={} min={}'.format(k, tuple(v.shape),v.max(),v.min()))
    for k, v in dataset.__getitem__(0).items():
        # print(k, v)
        log('  {}: shape={} max={} min={}'.format(k, tuple(v.shape),v.max(),v.min()))
    #my_sampler = MySampler(dataset, spec['batch_size'], 1, 4)
    if tag == 'train':
        loader = DataLoader(dataset, batch_size=spec['batch_size'],shuffle=(tag == 'train'), num_workers=12, pin_memory=False)
    else:
        loader = DataLoader(dataset, batch_size=spec['batch_size'],shuffle=(tag == 'train'), num_workers=2, pin_memory=False)
    #loader = DataLoader(dataset, batch_sampler = my_sampler, num_workers=16, pin_memory=True)
    

    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    
    val_loader2 = make_data_loader(config.get('val_dataset2'), tag='val')
    
    val_loader4 = make_data_loader(config.get('val_dataset4'), tag='val')
    return train_loader, val_loader2, val_loader4


def prepare_training():
    if (config.get('resume') is not None) and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'],map_location=torch.device('cpu'))
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        optimizer.param_groups[0]['lr'] = config['optimizer']['args']['lr']
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for e in range(1,epoch_start):
            lr_scheduler.step(e)
        print(epoch_start,optimizer.param_groups[0]['lr'])
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            cosine = CosineAnnealingLR(optimizer, config['epoch_max']-config['warmup_step_lr']['total_epoch'])
            lr_scheduler = GradualWarmupScheduler(optimizer,**config['warmup_step_lr'],after_scheduler=cosine)
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    param_count = utils.compute_num_params(model, text=True)
    for k, v in param_count.items():
        log('model: #{} params={:.1f}M'.format(k, v))
    
    # log(model)

    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model,optimizer, state, save_path,epoch=None, ker_loss_weight=1):
    model.train()
    loss_fn = nn.L1Loss().to('cuda')
    loss_fn_l2 = nn.MSELoss().to('cuda')
    # loss_fn_SSIM = pytorch_ssim.SSIM(window_size = 11)
    train_losses_img = []
    for i in range(25):
        train_losses_img.append(utils.Averager()) 

    train_loss = utils.Averager()
    train_loss_img = utils.Averager()
    train_loss_ker = utils.Averager()

    # train_loss_img0 = utils.Averager()
    # train_loss_img1 = utils.Averager()
    # train_loss_img2 = utils.Averager()
    # train_loss_img3 = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    train_weight = config['train_weight']
    pbar = tqdm(train_loader, leave=False, desc='train')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = (batch['inp'] - inp_sub) / inp_div
        if 'depth' in batch:
            input_depth = (batch['depth'] - inp_sub) / inp_div
        else:
            input_depth = None
            
        if 'label' in batch:
            input_seg = batch['label']
        else:
            input_seg = None
        kernel_gt = batch['kernel_map']
        gt = (batch['gt'] - gt_sub) / gt_div
        gt = gt.permute(0,3,1,2).contiguous()
        coord = batch['coord']
        scale = batch['scale']
        cell = batch['cell']
        if state == 'stage3V2_0_finetune':
            SR_pred, kernel, scale_encode= model(inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt, epoch=500)
        elif state == 'stage3V2_0' or state == 'stage3V2_1': 
            SR_pred, kernel, scale_encode= model(inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt, epoch=epoch)
        elif state == 'stage3V2_0_finetune_bench':
            SR_pred, kernel, scale_encode= model(inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt, epoch=1000, train_stage='bench')
        else:
            SR_pred, kernel, scale_encode= model(inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt)
        ker_loss = 0
        img_loss = 0
        if state == 'stage2':
            SR_pred = SR_pred
            img_loss = loss_fn(SR_pred, gt)
            train_loss_img.add(img_loss.item())
            loss = img_loss

        if state == 'stage3V2_0':
            # print(kernel.shape)
            # print(kernel_gt.shape)
            ker_loss = loss_fn(kernel, kernel_gt)
            train_loss_ker.add(ker_loss.item())
            img_loss = loss_fn(SR_pred, gt)
            train_loss_img.add(img_loss.item())
            if epoch < 300:
                loss = ker_loss
            else:
                loss = img_loss + ker_loss_weight*ker_loss
        if state == 'stage3V2_0_finetune':
            ker_loss = loss_fn(kernel, kernel_gt)
            train_loss_ker.add(ker_loss.item())
            img_loss = loss_fn(SR_pred, gt)
            train_loss_img.add(img_loss.item())
            loss = img_loss + ker_loss_weight*ker_loss
        if state == 'stage3V2_0_finetune_bench':
            # ker_loss = loss_fn(kernel, kernel_gt)
            # train_loss_ker.add(ker_loss.item())
            img_loss = loss_fn(SR_pred, gt)
            train_loss_img.add(img_loss.item())
            loss = img_loss

            
        train_loss.add(loss.item())
        # print(loss, loss.requires_grad)
        optimizer.zero_grad()
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        pbar.set_description('loss {:.4f} img_loss {:.4f} ker_loss {:.4f}'
                                 .format(train_loss.item(),train_loss_img.item(), train_loss_ker.item()))
        

    return train_loss.item(),  train_loss_img.item(), train_loss_ker.item(), train_losses_img

def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader2, val_loader4 = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    # load_G(config.get('kernel_encoder_path'), kernel_encoder, True)
    # load_G(config.get('kernel_encoder_path'), kernel_encoder, config.get('strict_load'))
    load_K(config.get('load_K'), model, config.get('strict_load'))
    # load_G(config.get('load_G'), model, True)
    load_G(config.get('load_G'), model, config.get('strict_load'))
    load_G_except_SFT(config.get('load_G_exSFT'), model, config.get('strict_load'))
    

    # liif_model_spec = torch.load(config.get('liif_pretrain'))['model']
    # liif_model = models.make(liif_model_spec, load_sd=True).cuda()
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    train_stage = config.get('state')
    ker_loss_weight = config.get('ker_loss_weight')
    print('ker_loss_weight', ker_loss_weight)
    max_val_v = 0
    max_val_mse_b = 1e9

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        train_loss, img_loss, ker_loss, img_losses= train(train_loader, model, optimizer, train_stage, save_path, epoch, ker_loss_weight=ker_loss_weight)
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']
        log_info.append('train: loss={:.4f}, img_loss={:.4f}, ker_loss={:.4f}, lr ={:.8f}'.format(train_loss, img_loss, ker_loss, lr))
        if train_stage == 'stage2_l2_3' or train_stage == '4_2_9_stage1' or train_stage == 'stage3V2_1' or train_stage == 'stage3V2_1_fintune' or train_stage == 'stage3V2_1_fintune_0':
            for i in range(4):
                log_info.append('train: img_loss{}={:.4f}'.format(i, img_losses[i].item()))

        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        version = config['version']
        if (epoch_save is not None) and ((epoch % epoch_save == 0) ):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        # if (epoch_val is not None) and ((epoch % epoch_val == 0) or (epoch == 1)):
            
        if (epoch_val is not None) and ((epoch % epoch_val == 0) or (epoch == 2000)):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
                
            if train_stage == 'stage3':
                PSNR_2, SSIM_2, LPIPS_2, MSE_B_2,_,_= eval_psnr(val_loader2, model_,
                data_norm=config['data_norm'],
                eval_type="div2k-2",
                eval_bsize=config.get('eval_bsize'),
                epoch=epoch,
                train_stage=train_stage,
                version=version
                )
                PSNR_4, SSIM_4, LPIPS_4, MSE_B_4,_,_ = eval_psnr(val_loader4, model_,
                data_norm=config['data_norm'],
                eval_type="div2k-4",
                eval_bsize=config.get('eval_bsize'),
                epoch=epoch,
                train_stage=train_stage,
                version=version,
                )
            elif train_stage=='stage3V2' or train_stage=='stage3V3' or train_stage=='stage3V2_0' or train_stage == 'stage3V2_0_finetune' or train_stage == 'stage3V2_0_finetune_freeze'or train_stage == 'stage3V2_l2' or train_stage =='stage3V2_SSIM' or train_stage =='stage3V2_1' or train_stage=='stage3V2_motion' or train_stage == 'stage3V2_1_fintune' or train_stage == 'stage3V2_1_fintune_0' or train_stage == 'stage3V2_1_finetune' or train_stage == 'stage3V2_1_finetune_fixKE':
                PSNR_2, SSIM_2, LPIPS_2, MSE_B_2, psnrs_2, mses_2= eval_psnr(val_loader2, model_,
                data_norm=config['data_norm'],
                eval_type="div2k-2",
                eval_bsize=config.get('eval_bsize'),
                epoch=epoch,
                train_stage=train_stage,
                version=version
                )
                PSNR_4, SSIM_4, LPIPS_4, MSE_B_4,psnrs_4, mses_4= eval_psnr(val_loader4, model_,
                data_norm=config['data_norm'],
                eval_type="div2k-4",
                eval_bsize=config.get('eval_bsize'),
                epoch=epoch,
                train_stage=train_stage,
                version=version
                )
                PSNR_4 = psnrs_2[-1].item()
                for j in range(len(psnrs_2)):
                    log_info.append('val2s: psnr2s={:.4f} mse2s={:.4f}'.format(psnrs_2[j].item(), mses_2[j].item()))
                for j in range(len(psnrs_4)):
                    log_info.append('val4s: psnr4s={:.4f} mse4s={:.4f}'.format(psnrs_4[j].item(), mses_4[j].item()))
            else:
                PSNR_2, SSIM_2, LPIPS_2, MSE_B_2,_,_= eval_psnr(val_loader2, model_,
                data_norm=config['data_norm'],
                eval_type="div2k-2",
                eval_bsize=config.get('eval_bsize'),
                epoch=epoch,
                train_stage=train_stage,
                version=version
                )
                PSNR_4, SSIM_4, LPIPS_4, MSE_B_4,_,_ = eval_psnr(val_loader4, model_,
                data_norm=config['data_norm'],
                eval_type="div2k-4",
                eval_bsize=config.get('eval_bsize'),
                epoch=epoch,
                train_stage=train_stage,
                version=version
                )

            log_info.append('val2: psnr={:.4f},SSIM={:.4f},LPIPS={:.4f},MSE={:.8f}'.format(PSNR_2,SSIM_2,LPIPS_2,MSE_B_2))
            log_info.append('val4: psnr={:.4f},SSIM={:.4f},LPIPS={:.4f},MSE={:.8f} '.format(PSNR_4,SSIM_4,LPIPS_4,MSE_B_4))
            
            writer.add_scalars('Metrics', {'PSNR': PSNR_4, 'SSIM': SSIM_4, 'LPIPS': LPIPS_4, 'MSE': MSE_B_4}, epoch)
            if PSNR_4 > max_val_v:
                max_val_v = PSNR_4
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
            if MSE_B_4 < max_val_mse_b:
                max_val_mse_b = MSE_B_4
                torch.save(sv_file, os.path.join(save_path, 'epoch-blurbest.pth'))
            # if MSE_B_4[-1].item() < max_val_mse_b:
            #     max_val_mse_b = MSE_B_4[-1].item()
            #     torch.save(sv_file, os.path.join(save_path, 'epoch-blur-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    print(save_path)
    main(config, save_path)
