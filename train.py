import time
import json
import wandb

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.amp import GradScaler, autocast
from einops import rearrange

from PIL import Image as PILImage
import os
import random

from util import get_mgrid, to_wandb_image, get_mem, sample, compute_psnr, preview_img
from siren_model import SirenModel
from tconfig import get_config
from wt import WT
from datasets import fetch_dataset, unpack
from validation import validation
from meta_sgd import MetaSGD


def forward(image, sample_size, preview_dict, build_graph, inner_steps):

    # print('TAKING NUM STEPS', inner_steps)

    if is_3d:
        udf_points, img_flat = image
        N = udf_points.shape[0]
    else:
        N, _, H, W = image.shape
        img_flat = rearrange(image, 'n c h w -> n (h w) c') # [N, H*W, C]
        coords_flat = get_mgrid(H, W, device) # [H*W, 2]
        coords_flat = coords_flat.unsqueeze(0).expand(N, -1, -1) # [N, H*W, 2]

    weight_offset = nn.Parameter(torch.zeros(size=(N, PI), device=device), requires_grad=True)
    for j in range(inner_steps + 1):
        last = (j == inner_steps)

        if is_3d: # # [N, ST, P_UDF, XYZ=3], [N, ST, P_UDF]
            use_coords = udf_points[:, j] # [N, P_UDF, 3]
            use_image = img_flat[:, j] # [N, P_UDF, 1]
        else:
            use_coords, use_image, indices = sample(sample_size, coords_flat, img_flat) # [N, 10000, 2], [N, 10000, 3]
            if (j == 0):
                preview_img(siren_rgb, coords_flat, weight_offset, indices, preview_dict, device, H, W)

        pred_rgbw = siren_rgb(use_coords, weight_offset=weight_offset) # [N, 10000, 3]
        assert pred_rgbw.shape[2] == conf['color_channels'], 'pred_rgbw should have C channels'
        loss = ((pred_rgbw - use_image) ** 2).mean(2) # [N, H*W]
        loss_summed = loss.mean(dim=1) # [N]

        if not last:
            grads_wrt_mods = torch.autograd.grad(loss_summed.sum(), [ weight_offset ], create_graph=build_graph)[0] # [N, TI]
            if build_graph:
                assert grads_wrt_mods.requires_grad, 'grads should require grad'

            # update weight offsets
            weight_offset = weight_offset - meta_sgd(j) * grads_wrt_mods # [N, WC]

            if not is_3d:
                preview_img(siren_rgb, coords_flat, weight_offset, indices, preview_dict, device, H, W)

        else:
            # compute the PSNR for the last step
            if is_3d:
                psnr = torch.zeros(N, device=device)
            else:
                psnr = compute_psnr(use_image, pred_rgbw)

    # loop over each output token (the amount of incoming values is constant except for first layer)
    vls = siren_rgb.sirens
    shapes = [ p.shape for p in vls ]
    l = []
    c = 0
    for shape in shapes:
        w = weight_offset[:, c:(c+shape.numel())]
        w = w.view(*([N] + list(shape)))
        w = w.detach() # speed up, no need to backprop is this is the case
        l.append(w)
        c += shape.numel()
    assert c == siren_rgb.num_params(), 'c should be equal to weight count but %d != %d' % (c, WC)
    # make pairs of 2, and skip first layer
    l = [ (l[i], l[i+1]) for i in range(2, len(l) - 1, 2) ]
    # logits = classifier(l) # [N, 10]
    return None, loss_summed.mean(), psnr


if __name__ == '__main__':
    
    conf = get_config()
    device = conf['device']
    project_name = "debug" if conf.get("debug") else "MWT-wx-implement"
    wandb.init(
        name=conf['name'],
        project=project_name,
        config=conf,
    )

    # if conf['auto_cast']:
    scaler = GradScaler(enabled=conf['auto_cast'])

    loader_train, loader_val_train, loader_val_val = fetch_dataset(conf)
    tot_batches = len(loader_train) * conf['epochs']

    param_groups_siren = []

    is_3d = conf['is_3d']
    siren_rgb = SirenModel(ch_in=(3 if is_3d else 2), ch_hiddens=conf['siren_dim'],
                            ch_out=conf['color_channels'], conf=conf, omega=conf['omega']).to(device)
    param_groups_siren += [ {'params': list(siren_rgb.parameters()), 'lr': conf['lr_siren_rgb']} ]
    PI = siren_rgb.pass_indices.shape[0]
    print('Passing number of indices:', PI)

    meta_sgd = MetaSGD(conf, PI).to(device)
    param_groups_siren += [ {'params': list(meta_sgd.parameters()), 'lr': conf['lr_sgd_lrs']} ]

    # classifier = build_classifier()

    # now merge and count params
    param_groups = (param_groups_siren)
    tp_siren = sum([ p.numel() for pg in param_groups_siren for p in pg['params'] ])

    opts = []
    scheds = []

    if conf['stages'] == 'simultaneous':
        # single sched and optimizer for both
        opts += [ torch.optim.AdamW(param_groups, lr=0, weight_decay=conf['weight_decay']) ]
        max_lrs = [ c['lr'] for c in param_groups ]
        scheds  += [ torch.optim.lr_scheduler.OneCycleLR(opts[0], pct_start=0.05, max_lr=max_lrs, total_steps=tot_batches) ]

    else:
        raise ValueError('unknown stage', conf['stages'])

    global_i = 0
    epoch = 0

    last_eval = time.time()
    val_data = {}
    # add num params as well
    val_data['params_siren'] = tp_siren

    gpu_mem = 0
    accumulated = {}

    checkpoint_dir = './checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # check if there exists checkpoint, if so, load it
    checkpoint_file = (checkpoint_dir + conf['name'] + '.pt')
    # if os.path.exists(checkpoint_file):
    #     print('[IMPORTANT] Found checkpoint, loading...')
    #     checkpoint = torch.load(checkpoint_file, weights_only=True)
    #     siren_rgb.load_state_dict(checkpoint['siren_rgb'])
    #     meta_sgd.load_state_dict(checkpoint['meta_sgd'])
    #     #opt.load_state_dict(checkpoint['opt'])
    #     #sched.load_state_dict(checkpoint['sched'])
    #     for i, o in enumerate(opts):
    #         o.load_state_dict(checkpoint['opts'][i])
    #     for i, s in enumerate(scheds):
    #         s.load_state_dict(checkpoint['scheds'][i])
    #     global_i = checkpoint['global_i']
    #     scaler.load_state_dict(checkpoint['scaler'])
    #     print('Loaded checkpoint, starting from global_i', global_i)

    first_epoch = True # after first epoch, we check gpu mem
    while True:

        """
        TRAINING
        """
        print('Starting new epoch of training...')
        epoch_start_time = time.time()

        for tp in loader_train:
            image, label = unpack(tp, conf, device)

            p_global = global_i / (tot_batches * len(opts))

            do_log = ((global_i + 1) % conf['interval_log'] == 0)
            do_preview = ((global_i + 1) % conf['interval_preview'] == 0)
            if do_preview:
                assert do_log, 'should log if previewing'

            preview_dict = {}
            with autocast(device_type=device, dtype=torch.float16, enabled=conf['auto_cast']):

                _, inner_loss, psnr_running = forward(image, conf['sample_size'],
                                    preview_dict if do_preview else None,
                                    build_graph=True,
                                    inner_steps=conf['inner_steps'])
                psnr_running = psnr_running.mean()


            # two-pass backward pass (we tried single pass and scaling cls_loss, but it was less stable)
            # scaler.scale(cls_loss).backward(retain_graph=True) # grads are scaled, but no problem as scaling is only multiplicative
            # cls_w = conf['cls_loss_weight']
            # for group in param_groups_siren:
            #     for p in group['params']:
            #         if p.grad is not None:
            #             p.grad *= cls_w
            scaler.scale(inner_loss).backward()

            # figure out which opt and sched based on global_i
            opt_ind = global_i // tot_batches
            if opt_ind >= len(opts):
                print('Finished training all optimizers, quitting...')
                exit(0)

            opt = opts[opt_ind]
            sched = scheds[opt_ind]

            # gradient clipping
            # if conf['auto_cast']:
            scaler.unscale_(opt)

            clip_val = 1.0
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], clip_val)
            
            # update step
            scaler.step(opt)
            scaler.update()

            sched.step()
            for opt in opts:
                opt.zero_grad()

            global_i += 1
            
            wandb_log = {
                'inner_loss': float(inner_loss),
                'done': float(p_global),
                'lr': float(sched.get_last_lr()[0]),
                'params_siren': tp_siren,
                'psnr_running': float(psnr_running),
            }

            if global_i % 100 == 0:
                print(wandb.run.name, wandb_log, flush=True)

            # accumulate
            for k, v in wandb_log.items():
                # check if this is a number
                if isinstance(v, (int, float)):
                    if k not in accumulated:
                        accumulated[k] = 0
                    accumulated[k] += v
                else:
                    raise Exception('should not log non-numeric values before this point, type was ' + str(type(v)))

            if do_log:
                # log averages
                accumulated = { k: v / conf['interval_log'] for k, v in accumulated.items() }
                accumulated['name'] = wandb.run.name

                if do_preview and not is_3d:
                    accumulated['img'] = to_wandb_image(rearrange(image[0], 'c h w -> h w c'), caption='input')
                    for k, ls in preview_dict.items():
                        if len(ls) > 0:
                            for i, img in enumerate(ls):
                                accumulated[f'{k}_{i}'] = to_wandb_image(img, caption=f'{k}_{i}')

                for k, v in val_data.items():
                    accumulated[k] = v

                wandb.log(accumulated)
                print(accumulated, flush=True)
                accumulated = {}

        epoch_end_time = time.time()

        if first_epoch:
            first_epoch = False
            gpu_mem = get_mem()
            val_data['gpu_mem_MiB'] = gpu_mem
            val_data['epoch_timings'] = []
            val_data['validation_timings'] = []

        epoch_took_time = (epoch_end_time - epoch_start_time) / 60.0 # in minutes
        val_data['epoch_timings'].append(epoch_took_time) # in minutes
    
        """
        VALIDATION
        """
        print('Running validation...!')

        val_start_time = time.time()
        validation(loader_val=loader_val_val, name='val', inner_steps=conf['inner_steps_test'],\
                    forward=forward, device=device, val_data=val_data, conf=conf)
        val_end_time = time.time()
        val_took_time = (val_end_time - val_start_time) / 60.0
        val_data['validation_timings'].append(val_took_time)

        validation(loader_val=loader_val_train, name='train', inner_steps=conf['inner_steps'],\
                    forward=forward, device=device, val_data=val_data, conf=conf)
        print('Validation done...!')

        """
        CHECKPOINTING
        """
        torch.save({
            'siren_rgb': siren_rgb.state_dict(),
            'meta_sgd': meta_sgd.state_dict(),
            #'opt': opt.state_dict(),
            #'sched': sched.state_dict(),
            'opts': [ o.state_dict() for o in opts ],
            'scheds': [ s.state_dict() for s in scheds ],
            'global_i': global_i,
            'scaler': scaler.state_dict(),
        }, checkpoint_file)

        # store config, together with validation data on disk
        result_folder = './results/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        data = { 'conf': conf, 'val_data': val_data, 'tot_steps': tot_batches * len(opts), 'current_step': global_i }
        # store as plain text json, name is the run name
        with open(result_folder + conf['name'] + '.json', 'w') as f:
            json.dump(data, f, indent=2)
            print('Stored validation data to disk.')


    print('Done, quitting...')