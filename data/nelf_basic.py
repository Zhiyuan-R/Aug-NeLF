import os, sys
import torch

class BaseNeLFDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, split='train', cam_id=False, rgb=True):
        
        super().__init__()
        
        self.split = split
        
        # Read metadata
        preprocess_path = os.path.join(root_dir, 'pre_process')
        with open(os.path.join(preprocess_path, 'meta.json'), 'r') as f:
            self.meta_dict = json.load(f)
            
        # Construct loaded filename
        rgbs_name, rays_name = 'rgbs_' + split, 'rays_' + split
        # Add extension name
        rgbs_name, rays_name = rgbs_name + '.npy', rays_name + '.npy' 
        
        # [N, H, W, ro+rd, 3]
        self.rays = np.load(os.path.join(preprocess_path, rays_name)) 
        
        # RGB files may not exist considering that exhibit ray exists
        if rgb:
            # [N, H, W, C]
            self.rgbs = np.load(os.path.join(preprocess_path, rgbs_name))
            
        # Basic attributes
        self.height = self.rays.shape[1]
        self.width = self.rays.shape[2]
        
        self.image_count = self.rays.shape[0]
        self.image_step = self.height * self.width

    def num_images(self):
        return self.image_count
    
    def height_width(self):
        return self.height, self.width
    
    def near_far(self):
        return self.meta_dict['near'], self.meta_dict['far']