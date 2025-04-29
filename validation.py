import os
import time
import torch
from torch.amp import GradScaler, autocast
from datasets import unpack

def validation(loader_val, name, inner_steps, forward, device, val_data, conf):
    is_3d = conf['is_3d']

    all_psnrs = []

    with autocast(enabled=conf['auto_cast'], device_type=device, dtype=torch.float16):
        timings = []
        inference_start = time.time()
        time_since_print = 0

        ijk = 0
        for tp in loader_val:
            ijk += 1
            image, label = unpack(tp, conf, device)
            _, _, psnrs = forward(image, conf['sample_size'], None, False, inner_steps=inner_steps)
            
            all_psnrs.append(psnrs.detach())

            if (time.time() - time_since_print) > 10:
                time_since_print = time.time()
                # print('validation scale', image.min(), image.max())
                print('time taken so far', (time.time() - inference_start) / 60.0, 'minutes', flush=True)
                print()

        
        torch.cuda.synchronize()
        inference_end = time.time()
        val_took = (inference_end - inference_start)
        
        psnr_avg = torch.cat(all_psnrs).mean()
        print(f'Validation {name} psnr: {psnr_avg}')

        val_data[f'psnr_{name}'] = float(psnr_avg)
