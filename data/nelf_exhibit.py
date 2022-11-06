import torch

from data import BaseNeLFDataset

class ExhibitNeRFDataset(BaseNeLFDataset):

    def __init__(self, root_dir, subsample=0):
        super().__init__(root_dir, split='exhibit', cam_id=False, rgb=False)

        self.rays = torch.from_numpy(self.rays).float()
        self.rays = self.rays.permute([0, 3, 1, 2, 4]) # [N, ro+rd, H, W, 3(+id)]

    def __len__(self):
        # return self.image_count * self.height * self.width
        return self.rays.shape[0]

    def __getitem__(self, i):
        return dict(rays = self.rays[i]) # [H, W, 3]