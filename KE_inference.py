import argparse
import os
import math
from functools import partial

import torch.nn.functional as F
import numpy as np
import yaml
import torch
import lpips
# from datasets.data import DataLoader

from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import model as models
import utils2
from utils2 import save_img,tensor2img, save_fake_img, save_numpy, channel_distribute

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=2)
    return pred

def eval_psnr(loader, model, kernel_encoder, version, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, epoch=0, train_stage=None):
    print(train_stage)
    model.eval()
    
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    scale = round(float(eval_type.split('-')[1]))
    PSNR_metric_fn = partial(utils2.calc_psnr, dataset='div2k', scale=scale)
    MSE_metric_fn = partial(utils2.calc_mse, dataset='div2k', scale=scale)
    SSIM_metric_fn = partial(utils2.calc_ssim, dataset='div2k', scale=scale)
    LPIPS_metric_fn = partial(utils2.calc_lpips, dataset='div2k', scale=scale)

    PSNR_res = utils2.Averager()
    PSNR_res2 = utils2.Averager()
    MSE_res2 = utils2.Averager()
    SSIM_res = utils2.Averager()
    LPIPS_res = utils2.Averager()
    PSNR_reses = []
    for i in range(25):
        PSNR_reses.append(utils2.Averager()) 
    pbar = tqdm(loader, leave=False, desc='val')
    lpips_model = lpips.LPIPS(net='alex') 
    idx = 0 
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = (batch['inp'] - inp_sub) / inp_div
        input_depth = (batch['depth'] - inp_sub) / inp_div
        kernel_gt = batch['kernel_map']
       
        gt = (batch['gt'] - inp_sub) / inp_div
        gt = gt.permute(0,3,1,2)
        input_seg = batch['label']
        coord = batch['coord']
        scale = batch['scale']
        cell = batch['cell']
        with torch.no_grad():
            SR_pred, kernel_pred, scale_encode= model(inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.data}")
        if train_stage == 'stage2_l2_pure_AE':
            gt = batch['gt'].permute(0,3,1,2)
           
            # SSIM_res.add(SSIM.item(), inp.shape[0])
            kernel_show_0 = kernel_pred[0]  #torch.sum(kernel_pred[0], dim=0).unsqueeze(0)
            q0 = kernel_pred[0].detach()
            q0 = q0.clamp(q0.quantile(0.03), q0.quantile(0.97))
            q0 = (q0 - q0.min())/(q0.max()- q0.min())
            q0 = (q0 - inp_sub)/(inp_div)
            q0 = q0.cpu().numpy()
            kernel_show_0 = kernel_show_0.clamp(kernel_show_0.quantile(0.03), kernel_show_0.quantile(0.97))
            kernel_show_0 = (kernel_show_0 - kernel_show_0.min())/(kernel_show_0.max()- kernel_show_0.min())
            # print(kernel_show_1.max(), kernel_show_1.min())
            save_numpy(q0, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data/"+ str(idx))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_gt/"+ str(idx) + "image_gt", tensor2img(gt))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual/"+ str(idx) + "kernel_pred", tensor2img(kernel_show_0))
        if train_stage == 'stage2_l2_AE':
            gt = batch['gt'].permute(0,3,1,2)
            SR_pred = SR_pred[0] * gt_div + gt_sub
            SR_pred = SR_pred.clamp(0,1)
            PSNR = PSNR_metric_fn(SR_pred, gt)
            # SSIM = SSIM_metric_fn(SR_pred, gt)
            LPIPS = LPIPS_metric_fn(SR_pred, gt, lpips_model)
            PSNR_res.add(PSNR.item(), inp.shape[0])
            # SSIM_res.add(SSIM.item(), inp.shape[0])
            kernel_show_0 = kernel_pred[0]  #torch.sum(kernel_pred[0], dim=0).unsqueeze(0)
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
            pbar.set_description('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} PSNR_B {:.4f}'
                                 .format(PSNR.item(),SSIM_res.item(),LPIPS_res.item(),PSNR_res.item()))
            q0 = kernel_pred[0][0].cpu().detach().numpy()
            # kernel_show_0 = kernel_show_0.clamp(kernel_show_0.quantile(0.05), kernel_show_0.quantile(0.95))
            kernel_show_0 = (kernel_show_0 - kernel_show_0.min())/(kernel_show_0.max()- kernel_show_0.min())

            
            print(q0.max(), q0.min())
            save_numpy(q0, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data/"+ str(idx))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_gt/"+ str(idx) + "image_gt", tensor2img(gt))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual/"+ str(idx) + "kernel_pred", tensor2img(kernel_show_0))
        if train_stage == 'stage2_l2_c15':
            gt = batch['gt'].permute(0,3,1,2)
            SR_pred = SR_pred[0] * gt_div + gt_sub
            SR_pred = SR_pred.clamp(0,1)
            PSNR = PSNR_metric_fn(SR_pred, gt)
            LPIPS = LPIPS_metric_fn(SR_pred, gt, lpips_model)
            PSNR_res.add(PSNR.item(), inp.shape[0])
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
            pbar.set_description('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} PSNR_B {:.4f}'
                                 .format(PSNR.item(),SSIM_res.item(),LPIPS_res.item(),PSNR_res.item()))
            q0 = kernel_pred[0].cpu().detach().numpy()
            # kernel_show_0 = kernel_pred.mean(dim=1, keepdim=False)
            print(q0.max(), q0.min())   
            kernel_show_0 = kernel_pred[0][0]
            kernel_show_0 = kernel_show_0.clamp(kernel_show_0.quantile(0.02), kernel_show_0.quantile(0.98))
            kernel_show_0 = (kernel_show_0 - kernel_show_0.min())/(kernel_show_0.max()- kernel_show_0.min())
            save_numpy(q0, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data/"+ str(idx))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_gt/"+ str(idx) + "image_gt", tensor2img(gt))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual/"+ str(idx) + "kernel_pred", tensor2img(kernel_show_0))
        if train_stage == 'stage2_c15':
            gt = batch['gt'].permute(0,3,1,2)
            SR_pred = SR_pred * gt_div + gt_sub
            SR_pred = SR_pred.clamp(0,1)
            PSNR = PSNR_metric_fn(SR_pred, gt)
            LPIPS = LPIPS_metric_fn(SR_pred, gt, lpips_model)
            PSNR_res.add(PSNR.item(), inp.shape[0])
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
            pbar.set_description('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} PSNR_B {:.4f}'
                                 .format(PSNR.item(),SSIM_res.item(),LPIPS_res.item(),PSNR_res.item()))
            q0 = kernel_pred[0].cpu().detach().numpy()
            # kernel_show_0 = kernel_pred.mean(dim=1, keepdim=False)  
            # kernel_show_0 = kernel_show_0.clamp(kernel_show_0.quantile(0.02), kernel_show_0.quantile(0.98))
            for j in range(15):
                kernel_show_0 = kernel_pred[0][j]
                kernel_show_0 = (kernel_show_0 - kernel_show_0.min())/(kernel_show_0.max()- kernel_show_0.min())
                save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual/"+ str(idx) +'_'+str(j)+ "kernel_pred", tensor2img(kernel_show_0))
            save_numpy(q0, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data/"+ str(idx))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_gt/"+ str(idx) + "image_gt", tensor2img(gt))
        if train_stage == 'stage2_l2':
            gt = batch['gt'].permute(0,3,1,2)
            SR_pred = SR_pred[0] * gt_div + gt_sub
            SR_pred = SR_pred.clamp(0,1)
            PSNR = PSNR_metric_fn(SR_pred, gt)
            # SSIM = SSIM_metric_fn(SR_pred, gt)
            LPIPS = LPIPS_metric_fn(SR_pred, gt, lpips_model)
            PSNR_res.add(PSNR.item(), inp.shape[0])
            # SSIM_res.add(SSIM.item(), inp.shape[0])
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
            pbar.set_description('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} PSNR_B {:.4f}'
                                 .format(PSNR.item(),SSIM_res.item(),LPIPS_res.item(),PSNR_res.item()))
            q0 = kernel_pred[0][0].cpu().detach().numpy()
            q = kernel_pred[1][0].cpu().detach().numpy()
            # channel_distribute(kernel_pred[0].cpu().detach(), "/gdata1/xiajy/model_save/vis/c3/test/kernel"+str(idx))
            # channel_distribute(gt[0].cpu().detach(), "/gdata1/xiajy/model_save/vis/c3/test/image_gt"+str(idx))
            # channel_distribute(kernel_gt[0].cpu().detach(), "/gdata1/xiajy/model_save/vis/c3/test")
            kernel_show_0 = kernel_pred[0][0]  #torch.sum(kernel_pred[0], dim=0).unsqueeze(0)
            kernel_show_0 = kernel_show_0.clamp(kernel_show_0.quantile(0.05), kernel_show_0.quantile(0.95))
            kernel_show_0 = (kernel_show_0 - kernel_show_0.min())/(kernel_show_0.max()- kernel_show_0.min())

            
            kernel_show_1 = kernel_pred[1][0]  #torch.sum(kernel_pred[0], dim=0).unsqueeze(0)
            kernel_show_1 = kernel_show_1 * gt_div + gt_sub
            # kernel_show = kernel_show.mean(dim=0, keepdim=False)
            kernel_show_1 = kernel_show_1.clamp(kernel_show_1.quantile(0.05), kernel_show_1.quantile(0.95))
            kernel_show_1 = (kernel_show_1 - kernel_show_1.min())/(kernel_show_1.max()- kernel_show_1.min())
            # print(kernel_show_1.max(), kernel_show_1.min())
            save_numpy(q0, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data0/"+ str(idx))
            save_numpy(q, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data/"+ str(idx))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_gt/"+ str(idx) + "image_gt", tensor2img(gt))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual0/"+ str(idx) + "kernel_pred", tensor2img(kernel_show_0))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual/"+ str(idx) + "kernel_pred", tensor2img(kernel_show_1))
            
        if train_stage == 'stage2':
            gt = batch['gt'].permute(0,3,1,2)
            inp = batch['inp']
            SR_pred = SR_pred * gt_div + gt_sub
            SR_pred = SR_pred.clamp(0,1)
            PSNR = PSNR_metric_fn(SR_pred, gt)
            SSIM = SSIM_metric_fn(SR_pred, gt)
            LPIPS = LPIPS_metric_fn(SR_pred, gt, lpips_model)
            PSNR_res.add(PSNR.item(), inp.shape[0])
            SSIM_res.add(SSIM.item(), inp.shape[0])
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
            pbar.set_description('PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} PSNR_B {:.4f}'
                                 .format(PSNR.item(),SSIM_res.item(),LPIPS_res.item(),PSNR_res.item()))
            q = kernel_pred[0].cpu().detach().numpy()
            # channel_distribute(kernel_pred[0].cpu().detach(), "/gdata1/xiajy/model_save/vis/c3/test/kernel"+str(idx))
            # channel_distribute(gt[0].cpu().detach(), "/gdata1/xiajy/model_save/vis/c3/test/image_gt"+str(idx))
            # channel_distribute(kernel_gt[0].cpu().detach(), "/gdata1/xiajy/model_save/vis/c3/test")
            kernel_show = kernel_pred[0]  #torch.sum(kernel_pred[0], dim=0).unsqueeze(0)
            
            kernel_show = kernel_show.clamp(kernel_show.quantile(0.05), kernel_show.quantile(0.95))
            kernel_show = (kernel_show - kernel_show.min())/(kernel_show.max()- kernel_show.min())
            # print(kernel_show.max(), kernel_show.min())

            # kernel_show = kernel_show * gt_div + gt_sub
            # kernel_show = kernel_show.mean(dim=0, keepdim=False)
            save_numpy(q, "/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/data/"+ str(idx))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_LR/"+ str(idx) + "image_LR", tensor2img(inp))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_SR/"+ str(idx) + "image_SR", tensor2img(SR_pred))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/image_gt/"+ str(idx) + "image_gt", tensor2img(gt))
            save_img("/gdata2/xiajy/dataset/depth/KE/" +str(version) + "/visual/"+ str(idx) + "kernel_pred", tensor2img(kernel_show))
            # save_img("/gdata1/xiajy/depth/kernel_encoder/" + str(idx)+"kernel_pred", tensor2img(kernel_pred))
            # save_fake_img("/gdata1/xiajy/depth/kernel_ecoder_sigmoid_test/visual/" + str(idx)+"kernel_pred", kernel_pred[0].cpu().detach())
            # save_img("/gdata1/xiajy/depth/kernel_encoder/" + str(idx)+"image_gt", tensor2img(gt))
        torch.cuda.empty_cache()
        idx += 1
    return PSNR_res.item(),SSIM_res.item(),LPIPS_res.item(), PSNR_res.item()
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['val_dataset4']
    array = []
    with np.load(spec['root_path_KernelWarehouse']) as npzfile:
        array = npzfile['arr_0']
    kernel_warehouse = torch.from_numpy(array)
    pca_matrix = torch.load(spec['pca_matrix_path'])[None]
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset,'KernelWarehouse': kernel_warehouse, 
                                                   'pca_matrix' :pca_matrix})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=1, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    if config['state'] == 'stage1':
        kernel_encoder_spec = torch.load(config['kernel_encoder'])['model']
        kernel_encoder = models.make(kernel_encoder_spec, load_sd=True).cuda()
    else:
        kernel_encoder = None
    
    PSNR, SSIM, LPIPS, PSNR_B = eval_psnr(loader, model, kernel_encoder, config.get('version'), 
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True,
        train_stage= config.get('state'))
    
    print('result: PSNR {:.4f},SSIM {:.4f}, lpips {:.4f} PSNR_B {:.4f}'.format(PSNR, SSIM,LPIPS, PSNR_B))








