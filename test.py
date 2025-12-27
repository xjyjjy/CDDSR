import argparse
import os
import math
from functools import partial

import torch.nn.functional as F
import numpy as np
import yaml
import torch
import lpips
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import make_coord

from thop import profile

import datasets
import model as models

from PIL import Image, ImageEnhance
import utils
from utils import save_img,tensor2img, save_fake_img
from utils import to_pixel_samples_grid 
# 74240 patch 64 31.2207
# 74253 patch 256 31.2519
# patch 31.0838
# all 32.45
def clip_test(img_lq, model, scale,input_depth, input_seg):
        all_flops = []
        sf = scale
        b, c, h, w = img_lq.size()
        tile = min(64, h, w)
        tile_overlap = 0
    
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, round(h*sf), round(w*sf)).type_as(img_lq)
        W = torch.zeros_like(E)
        print(E.shape)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile].cuda()
                input_depth_patch = input_depth[..., h_idx:h_idx+tile, w_idx:w_idx+tile].cuda()
                input_seg_patch = input_seg[..., h_idx:h_idx+tile, w_idx:w_idx+tile].cuda()
                target_size = (round(in_patch.shape[-2]*sf), 
                               round(in_patch.shape[-1]*sf))
                crop_hr = torch.ones(3,round(in_patch.shape[-2]*sf), round(in_patch.shape[-1]*sf)).float().cuda()
                hr_coord, hr_rgb = to_pixel_samples_grid(crop_hr.contiguous())
                cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32).cuda()
            
                with torch.no_grad():
                    out_patch, kernel_pred, scale_encode= model(in_patch, input_depth_patch, input_seg_patch, hr_coord.unsqueeze(0), scale, cell.unsqueeze(0), crop_hr.unsqueeze(0), None, "stage3_test")
        
                out_patch = out_patch[-1]
                # print("out_patch", out_patch.shape)
                # ih, iw = in_patch.shape[-2:]
                # shape = [in_patch.shape[0], round(ih * sf), round(iw * sf), 3]
                # out_patch = out_patch.view(*shape).permute(0, 3, 1, 2).contiguous()

                out_patch_mask = torch.ones_like(out_patch)

                E[..., round(h_idx*sf):round((h_idx+tile)*sf), round(w_idx*sf):round((w_idx+tile)*sf)].add_(out_patch)
                W[..., round(h_idx*sf):round((h_idx+tile)*sf), round(w_idx*sf):round((w_idx+tile)*sf)].add_(out_patch_mask)
        output = E.div_(W)
        #output = output.view(b, 3, -1).permute(0,2,1).contiguous()
        
        return output

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

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, epoch=0, train_stage='stage3', version=None, save_path=None):
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
    dataset = eval_type.split('-')[0]
    PSNR_metric_fn = partial(utils.calc_psnr, dataset=dataset, scale=scale)
    MSE_metric_fn = partial(utils.calc_mse, dataset=dataset, scale=scale)
    SSIM_metric_fn = partial(utils.calc_ssim, dataset=dataset, scale=scale)
    LPIPS_metric_fn = partial(utils.calc_lpips, dataset=dataset, scale=scale)

    PSNR_res = utils.Averager()
    PSNR_res2 = utils.Averager()
    MSE_res2 = utils.Averager()
    SSIM_res = utils.Averager()
    LPIPS_res = utils.Averager()
    PSNR_reses = []
    MSE_reses = []  
    for i in range(25):
        PSNR_reses.append(utils.Averager()) 
        MSE_reses.append(utils.Averager())
    pbar = tqdm(loader, leave=False, desc='val')
    lpips_model = lpips.LPIPS(net='alex') 
    idx = 0 
    for batch in pbar:
        for k, v in batch.items():
            if k == 'inp' or k == 'gt' or k == 'depth' or k == 'label' or k == 'coord' or k == 'scale' or k == 'cell':
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
        # kernel_code = torch.zeros((1,3,*kernel_gt.shape[-2:])).cuda()
        # print(batch['gt'].shape)
        gt = (batch['gt'] - inp_sub) / inp_div
        gt = gt.permute(0,3,1,2)
        # print(gt.shape)

        bs, c, h, w =gt.shape
        # if h > 1356 or w > 2040:
        #     idx += 1
        #     continue
        coord = batch['coord']
        scale = batch['scale']
        cell = batch['cell']
        # scale_max = 4
        # scale_num = max(scale/scale_max, 1)
        with torch.no_grad():
            SR_pred, kernel_pred, scale_encode= model(inp, input_depth, input_seg, coord, scale, cell, gt, kernel_gt, 'stage3_test', 1000)

      
        if train_stage == 'stage2':
            gt = batch['gt'].permute(0,3,1,2)
            SR_pred = SR_pred * gt_div + gt_sub
            SR_pred.clamp(0,1)
            PSNR = PSNR_metric_fn(SR_pred, gt)
            # SSIM = SSIM_metric_fn(SR_pred, gt)
            LPIPS = LPIPS_metric_fn(SR_pred, gt, lpips_model)
            PSNR_res.add(PSNR.item(), inp.shape[0])
            # SSIM_res.add(SSIM.item(), inp.shape[0])
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
       
        if train_stage == 'stage3V2_1' or train_stage == 'stage3V2_1_fintune' or train_stage == 'stage3V2_1_fintune_0' or train_stage== 'stage3V2_1_finetune' or train_stage == 'stage3V2_1_finetune_fixKE':
            gt = batch['gt'].permute(0,3,1,2)
            for i in range(0, len(SR_pred)):
                SR = SR_pred[i][0]* gt_div + gt_sub
                kernel = kernel_pred[i]
                SR = SR.clamp(0,1)
                PSNR = PSNR_metric_fn(SR, gt)
                # PSNR_res.add(PSNR.item(), inp.shape[0])
                MSE_blur_map = MSE_metric_fn(kernel, kernel_gt)
                MSE_reses[i].add(MSE_blur_map.item(), inp.shape[0])
                PSNR_reses[i].add(PSNR.item(), inp.shape[0])
        if train_stage == 'stage3V2' or train_stage == 'stage3V3' or train_stage == 'stage3V2_0' or train_stage == 'stage3V2_l2' or train_stage == 'stage3V2_SSIM' or train_stage == 'stage3V2_motion' or train_stage =='stage3V2_0_finetune' or train_stage == 'stage3V2_0_finetune_freeze':
            gt = batch['gt'].permute(0,3,1,2)
            for i in range(0, len(SR_pred)):
                SR = SR_pred[i] * gt_div + gt_sub
                # SR = SR_pred[i]
                kernel = kernel_pred[i]
                SR = SR.clamp(0,1)
                PSNR = PSNR_metric_fn(SR, gt)
                # PSNR_res.add
                # MSE_blur_map = MSE_metric_fn(kernel, kernel_gt)
                # MSE_reses[i].add(MSE_blur_map.item(), inp.shape[0])
                PSNR_reses[i].add(PSNR.item(), inp.shape[0])
            # print('PSNR', PSNR.item())
            LPIPS = LPIPS_metric_fn(SR, gt, lpips_model)
            LPIPS_res.add(LPIPS.item(), inp.shape[0])
            sr_for_save = SR_pred[-1] * gt_div + gt_sub
            sr_for_save = sr_for_save.clamp(0,1)
            # print(save_path)
            if save_path is not None:
                save_img(save_path + str(idx), tensor2img(sr_for_save))
    

        idx += 1
        # print(PSNR.item())
        # if idx != len(loader):
        #     del SR_pred, kernel_pred, scale_encode
        if verbose or True:
            pbar.set_description('PSNR_0 {:.4f}PSNR_1 {:.4f}  LPIPS {:.4f} '
                                 .format(PSNR_reses[0].item(),PSNR_reses[1].item(),LPIPS_res.item()))
        torch.cuda.empty_cache()
    # print('gt', gt[0,0,120,120])
    # print('scale_encode', scale_encode.max(), scale_encode.min())
    if train_stage == 'stage3V2' or train_stage == 'stage3V3' or train_stage =='stage3V2_0' or train_stage == 'stage3V2_0_finetune' or train_stage == 'stage3V2_0_finetune_freeze'or train_stage =='stage3V2_l2' or train_stage == 'stage3V2_SSIM' or train_stage == 'stage3V2_1' or train_stage == 'stage3V2_motion' or train_stage == 'stage3V2_1_fintune' or train_stage =='stage3V2_1_fintune_0' or train_stage == 'stage3V2_1_finetune' or train_stage == 'stage3V2_1_finetune_fixKE':
        return PSNR_res.item(),SSIM_res.item(),LPIPS_res.item(), MSE_res2.item(),PSNR_reses[0:len(SR_pred)], MSE_reses[0:len(SR_pred)]
    else:
        return PSNR_res.item(),SSIM_res.item(),LPIPS_res.item(), MSE_res2.item(),PSNR_reses[0:0], MSE_reses[0:0]
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['val_dataset']
    if 'save_path' in config:
        save_path = config['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = None
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
    eval_type = config['eval_type']
    if config.get('train_stage') is not None:
        train_stage = config.get('train_stage')
        PSNR_2, SSIM_2, LPIPS_2, MSE_B_2,psnrs_2,mses_2= eval_psnr(loader, model,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                epoch=0,
                train_stage=train_stage,
                version=None,
                save_path=save_path
                )
        for j in range(len(psnrs_2)):
            print('val2s: psnr2s={:.4f} mse2s={:.4f}'.format(psnrs_2[j].item(), mses_2[j].item()))
                
    else:
        train_stage = None
        PSNR_2, SSIM_2, LPIPS_2, MSE_B_2, psnrs_2, mses_2= eval_psnr(loader, model,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                epoch=0,
                # train_stage=train_stage,
                version=None,
                save_path = None
                )
    print('result: PSNR {:.4f},SSIM {:.4f}, lpips {:.4f} PSNR_B {:.4f}'.format(PSNR_2, SSIM_2,LPIPS_2, PSNR_2))








