'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import json
import numpy as np
import torch


class NormalizeInverse(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

denormalize_rgb = transforms.Compose([
    NormalizeInverse(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
    transforms.ToPILImage(),
])

normalize_gripper = transforms.Normalize(mean=[0.03228], std=[0.01])

denormalize_gripper = NormalizeInverse(mean=[0.03228], std=[0.01])


def build_predicates(objects, unary, binary):
    pred_names = [pred % obj for pred in unary for obj in objects]
    obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
    obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
    pred_names += [pred % (o1, o2) for pred in binary for o1, o2 in zip(obj1, obj2)]
    return pred_names


class CLEVRDataset(Dataset):
    def __init__(self, scene_file, obj_file, max_nobj, rand_patch):
        self.obj_file = obj_file
        self.obj_h5 = None
        self.scene_file = scene_file
        self.scene_h5 = None
        with h5py.File(scene_file) as scene_h5:
            self.scenes = list(scene_h5.keys())
        self.max_nobj = max_nobj
        self.rand_patch = rand_patch

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(self.obj_file)
        if self.scene_h5 is None:
            self.scene_h5 = h5py.File(self.scene_file)

        scene = self.scene_h5[self.scenes[idx]]
        img = normalize_rgb(Image.open(BytesIO(scene['image'][()])).convert('RGB'))

        objects = scene['objects'][()].decode().split(',')
        obj_patches = []
        for obj in objects:
            patch_idx = 0
            if self.rand_patch:
                patch_idx = torch.randint(len(self.obj_h5[obj]), ()).item()
            patch = normalize_rgb(Image.open(BytesIO(self.obj_h5[obj][patch_idx])))
            obj_patches.append(patch)
        for _ in range(len(obj_patches), self.max_nobj):
            obj_patches.append(torch.zeros_like(obj_patches[0]))
        obj_patches = torch.stack(obj_patches)

        relations, mask = [], []
        ids = np.arange(self.max_nobj)
        for relation in scene['relations']:
            for k in range(1, self.max_nobj):
                for i, j in zip(ids, np.roll(ids, -k)):
                    if i >= len(objects) or j >= len(objects):
                        relations.append(0)
                        mask.append(0)
                    else:
                        relations.append(relation[i][j])
                        mask.append(relation[i][j] != -1)
        relations = torch.tensor(relations).float()
        mask = torch.tensor(mask).float()

        return img, obj_patches, relations, mask


class LeonardoDataset(Dataset):
    def __init__(
        self, data_dir, split, predicates, obj_file, colors=None,
        randpatch=True, view=1, randview=True, gripper=False,
        img_size=(224,224)
    ):
        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)
        with h5py.File(f'{data_dir}/{split}.h5') as data:
            all_predicates = data['predicates'][()].decode().split('|')
            pred_ids = {pred: i for i, pred in enumerate(all_predicates)}
            self.pred_ids = [pred_ids[pred] for pred in predicates]

        self.data_dir = data_dir
        self.split = split
        self.h5 = None

        self.colors = colors
        self.obj_file = obj_file
        self.obj_h5 = None
        self.randpatch = randpatch

        self.view = view
        self.randview = randview

        self.gripper = gripper
        self.img_size = img_size

    def __len__(self):
        return self.cum_n_frames[-1]

    def load_h5(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(f'{self.data_dir}/{self.split}.h5', 'r')
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(f'{self.data_dir}/{self.obj_file}', 'r')
        # Get H5 file index and frame index
        file_idx = np.argmax(idx < self.cum_n_frames)
        data = self.h5[self.sequences[file_idx]]
        frame_idx = idx
        if file_idx > 0:
            frame_idx = idx - self.cum_n_frames[file_idx - 1]
        return data, frame_idx

    def get_rgb(self, data, idx):
        v = torch.randint(self.view, ()).item() if self.randview else self.view
        rgb = Image.open(BytesIO(data[f'rgb{v}'][idx])).convert('RGB')
        return normalize_rgb(rgb.resize(self.img_size))

    def get_patches(self, colors):
        obj_patches = []
        for color in colors:
            patch_idx = 0
            if self.randpatch:
                patch_idx = torch.randint(len(self.obj_h5[color]), ()).item()
            patch = Image.open(BytesIO(self.obj_h5[color][patch_idx]))
            obj_patches.append(normalize_rgb(patch))
        return torch.stack(obj_patches)

    def get_gripper(self, data, idx):
        gripper = data['gripper'][idx].astype('float32')
        gripper = torch.from_numpy(gripper).reshape(1, 1, 1)
        return normalize_gripper(gripper).squeeze()

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)

        # Load predicates from H5 file
        predicates = data['logical'][frame_idx][self.pred_ids]

        # Load RGB from H5 file
        rgb = self.get_rgb(data, frame_idx)

        # Load object patches
        colors = self.colors
        if colors is None:
            colors = data['colors'][()].decode().split(',')
        obj_patches = self.get_patches(colors)

        # Load gripper state from H5 file
        gripper = self.get_gripper(data, frame_idx) if self.gripper else 0
        
        return rgb, obj_patches, gripper, predicates


class RegressionDataset(LeonardoDataset):
    def __init__(
        self, data_dir, split, obj_file, colors=None, objects=None,
        randpatch=True, view=1, randview=True, ee=False, dist=False,
        img_size=(224,224)
    ):
        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)

        self.data_dir = data_dir
        self.split = split
        self.h5 = None

        self.colors = colors
        self.objects = objects
        self.obj_file = obj_file
        self.obj_h5 = None
        self.randpatch = randpatch

        self.view = view
        self.randview = randview

        self.ee = ee
        self.dist = dist
        self.img_size = img_size

    def __len__(self):
        return self.cum_n_frames[-1]

    def get_ee_obj_xyz(self, data, idx, objects):
        xyz = torch.stack([torch.from_numpy(
            data[f'{obj}_pose'][idx][:3, 3] - data['ee_pose'][idx][:3, 3]
        ) for obj in objects])
        return xyz

    def get_obj_obj_xyz(self, data, idx, objects):
        obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
        obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
        xyz = torch.stack([torch.from_numpy(
            data[f'{o2}_pose'][idx][:3, 3] - data[f'{o1}_pose'][idx][:3, 3]
        ) for o1, o2 in zip(obj1, obj2)])
        return xyz

    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)

        # Load RGB from H5 file
        rgb = self.get_rgb(data, frame_idx)

        # Load object patches
        colors = self.colors
        if colors is None:
            colors = data['colors'][()].decode().split(',')
        obj_patches = self.get_patches(colors)

        # Load regression targets
        objects = self.objects
        if objects is None:
            objects = [f'object{i:02d}' for i in range(len(colors))]
        if self.ee:
            # End effector to object center
            target = self.get_ee_obj_xyz(data, frame_idx, objects)
        else:
            # Object center to object center
            target = self.get_obj_obj_xyz(data, frame_idx, objects)
        if self.dist:
            target = target.norm(dim=-1, keepdim=True).float()
        else:
            target = torch.nn.functional.normalize(target).float()

        return rgb, obj_patches, target
