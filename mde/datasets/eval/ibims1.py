import os
import torch
from adict import adict

from mde.datasets.dataset import _Dataset


class IBIMS1Dataset(_Dataset):
    def __init__(self,
                 root,
                 size=None,
                 near=1e-2,
                 far=50.0,
                 partition="test",
                 eval_fullres=False):
        super().__init__(root, domain="depth", size=size)
        with open(os.path.join(root, 'imagelist.txt'), 'r') as f:
            self.samples = list(map(lambda s: s.strip(), f.readlines()))
        # clip space
        self.name = "IBIMS1"
        self.near = near
        self.far = far
        self.eval_fullres = eval_fullres

    def __len__(self):
        return len(self.samples)

    def load_mask(self, image_name, mask_type):
        mask_path = os.path.join(self.root, 'mask_'+mask_type, image_name + '.png')
        if os.path.isfile(mask_path):
            mask = _Dataset.read_target(mask_path, size=self.size if not self.eval_fullres else None).long()
        else:
            size = self.size if not self.eval_fullres else (480, 640)
            mask = torch.zeros(*size, dtype=torch.long)

        txt_path = os.path.join(self.root, 'mask_'+mask_type, image_name + '.txt')
        planes = torch.zeros(0, 7, dtype=torch.float32)
        if os.path.isfile(txt_path):
            with open(txt_path, 'r') as f:
                planes = torch.as_tensor(list(map(lambda x: list(map(float, x.strip().split(','))), f.readlines())))
        return mask, planes

    def __getitem__(self, item):
        image_name = self.samples[item]

        image_path = os.path.join(self.root, "rgb", image_name + '.png')
        image = _Dataset.read_image(image_path, size=self.size)

        depth_path = os.path.join(self.root, "depth", image_name + '.png')
        depth = _Dataset.read_target(
            depth_path,
            size=self.size if not self.eval_fullres else None,
            scale=65535.0 / 50.0
        )

        # valid pixels
        mask_invalid = _Dataset.read_target(
            os.path.join(self.root, "mask_invalid", image_name + ".png"),
            size=self.size if not self.eval_fullres else None
        ) > 0
        mask_transparent = _Dataset.read_target(
            os.path.join(self.root, "mask_transp", image_name + ".png"),
            size=self.size if not self.eval_fullres else None
        ) > 0
        mask = (depth > self.near) & (depth <= self.far) & mask_invalid & mask_transparent

        # Read and scale calibration if needed
        with open(os.path.join(self.root, 'calib', image_name + '.txt'), 'r') as f:
            fx, fy, cx, cy = list(map(float, f.readlines()[0].split(',')))
            if not self.eval_fullres:
                fx = fx / 640 * self.size[1]
                fy = fy / 480 * self.size[0]
                cx = cx / 640 * self.size[1]
                cy = cy / 480 * self.size[0]
            camera = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)

        # Depth edges map
        edges_path = os.path.join(self.root, 'edges', image_name + '.png')
        edges_valid = False
        if os.path.isfile(edges_path):
            edges_valid = True
            edges = _Dataset.read_target(edges_path, size=self.size if not self.eval_fullres else None).bool()
        else:
            edges = torch.zeros_like(mask)

        # planar regions masks and their spatial orientations
        mask_table, table_planes = self.load_mask(image_name, 'table')
        mask_floor, floor_planes = self.load_mask(image_name, 'floor')
        mask_walls, wall_planes = self.load_mask(image_name, 'wall')

        metadata = adict(image_path=image_path,
                         depth_path=depth_path,
                         domain=self.domain,
                         name=self.name,
                         range=(self.near, self.far),
                         edges_valid=edges_valid)

        return adict(image=image.clamp(0, 1.0),
                     target=depth,
                     mask=mask,
                     edges=edges,
                     camera=camera,

                     mask_table=mask_table,
                     table_planes=table_planes,
                     mask_floor=mask_floor,
                     floor_planes=floor_planes,
                     mask_walls=mask_walls,
                     wall_planes=wall_planes,

                     metadata=metadata)
