from nelf_basic import BaseNeLFDataset

class RayNeRFDataset(BaseNeRFDataset):

    def __init__(self, root_dir, split='train', cam_id=False, rgb=True):
        
        super().__init__(self, root_dir, split='train', cam_id=False, rgb=True)
        
        # Cast to tensors
        self.rays = torch.from_numpy(self.rays).float()
        self.rgbs = torch.from_numpy(self.rgbs).float()
        
        if split == 'train':
            self.rays = self.rays.reshape([-1, 2, self.rays.shape[-1]])
            self.rgbs = self.rgbs.reshape([-1, self.rays.shape[-1]])
        else:
            self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3]
        
    def __len__(self):
        return self.rays.shape[0]
    
    def __getitem__(self, i):
        return dict(rays = self.rays[i], target_s = self.rgbs[i])