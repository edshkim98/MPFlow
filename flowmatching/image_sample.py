"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import glob
import yaml
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Guided Diffusion imports
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import IQTDataset
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.condition_methods import get_conditioning_method

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data_custom(data_loader):
    while True:
        yield from data_loader

def main():
    set_seed(42)

    # Load Configs
    # Note: Ensure this path is correct for your environment
    config_path = '/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/flowmatching/configs_kspace.yaml'
    with open(config_path) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    # Add 'flow_matching' to config if passing it to creation is necessary, 
    # otherwise we handle it in the sampling loop.
    model, diffusion = create_model_and_diffusion(
        configs=configs,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Load Model Checkpoint
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print('Using device:', device)

    # Setup Paths
    save_path = '/cluster/project0/IQT_Nigeria/skim/fm_multimodal_t100_lr1_results_60k_x4kspace_noz/'
    
    # [FILTERING FILES LOGIC PRESERVED]
    lst_files = [
        '996782', '995174', '994273', '993675', '992774', '992673', '991267'
    ]
 
    data_dir = '/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim/9*'
    files = glob.glob(data_dir + '/T1w/T2w_acpc_dc_restore_brain.nii.gz')
    files_new = []
    for f in files:
        if f.split('/')[-3] in lst_files:
            files_new.append(f)
    files = files_new

    # Dataset
    dataset = IQTDataset(files, configs=configs, return_id=configs['data']['return_id'])
    print(f"Files: {len(files)} Dataset size: {len(dataset)}")
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    # Prepare Operator and noise
    measure_config = configs['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
 
    # Output Directories
    save_dir = '/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/results/'
    out_path = os.path.join(save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
           
    # Prepare conditioning method
    cond_config = configs['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {configs['conditioning']['method']}")

    logger.log("sampling...")
    all_images = []
    ys = []
    refs = []
    time_lst = []
    
    for i, (ref_img, img_mm, data_dict) in tqdm.tqdm(enumerate(data)):
        print(f"{i}/{len(data)}")
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
            
        ref_img = ref_img.to(device)
        fname_curr, slice_curr = int(data_dict['file_id'][0]), str(data_dict['slice_idx'].numpy()[0])
        print(fname_curr, slice_curr)
        
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Handle Skip Timesteps (SDEdit / Refinement)
        if configs['skip_timestep']:
            # For Flow Matching, this would be your starting point t > 0
            skip_x0 = y_n.clone().to(device) 
        else:
            skip_x0 = None

        if configs['multimodal']:
            img_mm = img_mm.to(device)
        else:
            img_mm = None

        # -----------------------------------------------------------
        # SELECT SAMPLING FUNCTION
        # -----------------------------------------------------------
        if args.flow_matching:
            # New Flow Matching path
            # Note: Ensure `flow_matching_sample_loop` is defined in GaussianDiffusion
            start = time.time()
            sample = diffusion.flow_matching_sample_loop(
                model=model,
                measurement=y_n.to(torch.float32),
                measurement_cond_fn=measurement_cond_fn,
                multimodal=img_mm,
                operator=[operator, cond_config['params']],
                clip_denoised=args.clip_denoised,
                shape=(args.batch_size, 1, args.image_size, args.image_size),
                noise=skip_x0 if configs['skip_timestep'] else None, # Start from noisy img if skipping
                model_kwargs=model_kwargs,
                steps=args.steps, # Use the new steps argument
                device=device,
                multi_seed=False,
                progress=False,
                solver='euler'
            )
            end = time.time()
        else:
            # Original Diffusion path
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            start = time.time()
            sample = sample_fn(
                model=model,
                shape=(args.batch_size, 1, args.image_size, args.image_size),
                measurement=y_n.to(torch.float32),
                measurement_cond_fn=measurement_cond_fn, # Disabled as per original script logic
                multimodal=img_mm,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                skip_timesteps=configs['skip_timestep'],
                skip_x0=skip_x0,
                line_search=configs['line_search'],
            )
            end = time.time()
        # -----------------------------------------------------------

        print("Inf time: ", end-start)
        time_lst.append(end-start)
        
        sample = sample.contiguous()
        all_images.append(sample.detach().cpu().numpy()) 
        refs.append(ref_img.detach().cpu().numpy())
        ys.append(y_n.detach().cpu().numpy())
        print("One image done!")
         
        if data_dict is not None:
            for j in range(args.batch_size):
                if not os.path.exists(f'{save_path}/{data_dict["file_id"][j]}'):        
                    os.makedirs(f'{save_path}/{data_dict["file_id"][j]}')
                np.save(f'{save_path}/{data_dict["file_id"][j]}/pred_{data_dict["slice_idx"][j]}_axial.npy', 
                       sample[j].detach().cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/gt_{data_dict["slice_idx"][j]}_axial.npy', 
                       ref_img[j].detach().cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/lr_{data_dict["slice_idx"][j]}_axial.npy', 
                       y[j].detach().cpu().numpy())
#                np.save(f'{save_path}/{data_dict["file_id"][j]}/gt_{data_dict["slice_idx"][j]}_axial.npy', sample[j].cpu().numpy())
     
    time_lst = np.array(time_lst)
    print("Mean time: ", np.mean(time_lst))
    print("Std time: ", np.std(time_lst))
    print("Saving the results in Numpy")
   
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        # New Arguments for Flow Matching
        flow_matching=True,  # Set True to use flow matching sampler
        steps=100,             # Number of ODE steps (defaults to 50, usually sufficient for FM)
        model_path='./logs_large_flowmatching_zero2two_t2/model060000.pt',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
