import configargparse
import os
import torch
import numpy as np
from tqdm import tqdm
import json

from data import load_blender_data
from utils import get_persp_intrinsic, get_persp_rays

def create_arg_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', is_config_file=True, help='Path to config file')
    parser.add_argument('--data_path', '--datadir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--data_type', '--dataset_type', type=str, required=True,
                        help='Specify the type of the dataset',
                        choices=['blender'])
    parser.add_argument('--half_res', action='store_true', default=False,
                        help='Load half-resolution (400x400) images instead of full resolution (800x800). \
                        Only for blender dataset.')
    parser.add_argument('--white_bkgd', action='store_true', default=False,
                        help='Render synthetic data on white background. Only for blender/LINEMOD dataset')
    parser.add_argument('--test_skip', type=int, default=8, 
                        help='will load 1/N images from test/val sets. Only for large datasets like blender/LINEMOD/deepvoxels.')
    
    return parser

if __name__ == '__main__':
    
    parser = create_arg_parser()
    args =  parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print('Dataset path not exists:', args.data_path)
        exit(-1)
        
    output_path = os.path.join(args.data_path, 'pre_process')
    os.makedirs(output_path, exist_ok=True)
    
    K = None # intrinsic matrix; take up some space
    
    if args.data_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.data_path, args.half_res, args.test_skip)
    print('Loaded blender ----> Done!')
    i_train, i_val, i_test = i_split
    
    near = 2.
    far = 6. 
    
    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]
    
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    if K is None:
        K = get_persp_intrinsic(H, W, focal)
    print('Intrinsic matrix:', K)
    print('Train/valid/test split', i_train, i_val, i_test)
    
    print('Calculating train/valid/test rays ...')

    rays = torch.stack([get_persp_rays(H, W, K, torch.tensor(p)) for p in tqdm(poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
    rays = rays.permute([0, 2, 3, 1, 4]).numpy().astype(np.float32) # [N, H, W, ro+rd, 3]
    print('Done.', rays.shape)

    print('Splitting train/valid/test rays ...')
    rays_train, rgbs_train = rays[i_train], images[i_train]
    rays_val, rgbs_val = rays[i_val], images[i_val]
    rays_test, rgbs_test = rays[i_test], images[i_test]
    
    print('Calculating exhibition rays ...')
    rays_exhibit = torch.stack([get_persp_rays(H, W, K, torch.tensor(p)) for p in tqdm(render_poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
    rays_exhibit = rays_exhibit.permute([0, 2, 3, 1, 4]).numpy().astype(np.float32) # [N, H, W, ro+rd, 3]
    print('Done.', rays_exhibit.shape)
    
    print('Training set:', rays_train.shape, rgbs_train.shape)
    print('Validation set:', rays_val.shape, rgbs_val.shape)
    print('Testing set:', rays_test.shape, rgbs_test.shape)
    print('Exhibition set:', rays_exhibit.shape)
    
    print('Saving to: ', output_path)
    np.save(os.path.join(output_path, 'rays_train.npy'), rays_train)
    np.save(os.path.join(output_path, 'rgbs_train.npy'), rgbs_train)

    np.save(os.path.join(output_path, 'rays_val.npy'), rays_val)
    np.save(os.path.join(output_path, 'rgbs_val.npy'), rgbs_val)

    np.save(os.path.join(output_path, 'rays_test.npy'), rays_test)
    np.save(os.path.join(output_path, 'rgbs_test.npy'), rgbs_test)

    np.save(os.path.join(output_path, 'rays_exhibit.npy'), rays_exhibit)
    
        # Save meta data
    meta_dict = {
        'H': H, 'W': W, 'focal': float(focal),
        'near': float(near), 'far': float(far),

        'i_train': i_train.tolist(), 'i_val': i_val.tolist(), 'i_test': i_test.tolist(),

        'half_res': args.half_res, 'white_bkgd': args.white_bkgd,
        'test_skip': args.test_skip,
    }
    print("Meta data:", meta_dict)
    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta_dict, f)