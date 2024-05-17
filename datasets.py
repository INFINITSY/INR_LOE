import os
import torch
import torchvision
import numpy as np
import csv
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2, max=1.0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-max, max, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class LatticeDataset(torch.utils.data.Dataset):

    def __init__(self, image_shape=(32, 32)):
        super().__init__()

        self.mgrid = self.get_2d_mgrid(image_shape)

    def get_2d_mgrid(self, shape):
        pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1).astype(np.float32)

        # normalize pixel coords onto [-1, 1]
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
        pixel_coords -= 0.5
        pixel_coords *= 2.
        # flatten 
        pixel_coords = torch.tensor(pixel_coords).view(-1, 2)

        return pixel_coords

    def __len__(self):
        return self.mgrid.shape[0]
    
    def __getitem__(self, idx):
        return self.mgrid[idx], torch.tensor(idx, dtype=torch.int64)
    

class TransposeTransform:

    def __init__(self, in_fmt='CHW', out_fmt='HWC'):
        self.order = [in_fmt.index(c) for c in out_fmt]

    def __call__(self, img):
        return img.permute(self.order)


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root, split, subset=-1, downsampled_size=None, patch_size=None):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val']

        self.img_dir = os.path.join(root, 'img_align_celeba')
        self.img_channels = 3
        self.file_names = []

        with open(os.path.join(root, 'list_eval_partition.txt'), newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in rowreader:
                # if split == 'train' and row[1] == '0':
                if split == 'train' and row[1] == '0':
                    self.file_names.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.file_names.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.file_names.append(row[0])
        if isinstance(subset, int):
            if subset > 0:
                self.file_names = self.file_names[:subset]
        elif isinstance(subset, list):
            self.file_names = [self.file_names[i] for i in subset]

        self.downsampled_size = downsampled_size if downsampled_size is not None else (178, 178)
        self.patch_size = patch_size if patch_size is not None else self.downsampled_size

    @property
    def num_patches_per_img(self):
        return (self.downsampled_size[0] // self.patch_size[0]) * (self.downsampled_size[1] // self.patch_size[1])

    @property
    def num_images(self):
        return self.num_patches_per_img * len(self.file_names)

    @property
    def num_channels(self):
        return 3

    @property
    def image_size(self):
        return self.patch_size

    @property
    def full_image_size(self):
        return self.downsampled_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches_per_img
        patch_idx = idx % self.num_patches_per_img
        path = os.path.join(self.img_dir, self.file_names[img_idx])
        assert os.path.exists(path), 'Index does not specify any images in the dataset'
        
        img = Image.open(path)

        width, height = img.size  # Get dimensions

        s = min(width, height)
        left = (width - s) / 2
        top = (height - s) / 2
        right = (width + s) / 2
        bottom = (height + s) / 2
        img = img.crop((left, top, right, bottom))

        if self.downsampled_size != img.size:
            img = img.resize(self.downsampled_size)

        img = np.asarray(img).astype(np.float32) / 255.

        # crop patch size
        if self.num_patches_per_img != 1:
            num_patches_per_row = self.downsampled_size[0] // self.patch_size[0]  # width
            row_idx, col_idx = patch_idx // num_patches_per_row, patch_idx % num_patches_per_row
            y, x = row_idx * self.patch_size[1], col_idx * self.patch_size[0]
            img = img[y:y+self.patch_size[1], x:x+self.patch_size[0]]

        # permute to CHW
        img = np.transpose(img, (2, 0, 1))

        in_dict = {
            'idx': torch.tensor(idx, dtype=torch.int64), 
            'coords': get_mgrid(self.downsampled_size[0]) 
            }
        gt_dict = {'img': torch.from_numpy(img)}

        return in_dict, gt_dict
    

class CIFAR10Dataset(torch.utils.data.Dataset):

    def __init__(self, root, split='train', subset=-1, class_label=5):
        super().__init__()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # TransposeTransform(in_fmt='CHW', out_fmt='HWC')
        ])
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=(split == 'train'),
                download=True, transform=transforms)
        # only keep all the dog images
        if class_label >= 0:
            self.images = [(img, label) for img, label in self.cifar10 if label == class_label]
        else:
            # keep all images
            self.images = self.cifar10
        if subset > 0:
            subset_idx = np.linspace(0, len(self.images)-1, subset, dtype=np.int64)
            self.images = torch.utils.data.Subset(self.images, subset_idx)

    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_channels(self):
        return 3

    @property
    def image_size(self):
        return (32, 32)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # return torch.mean(self.images[idx][0], dim=-1, keepdim=True), torch.tensor(idx, dtype=torch.int64)
        return self.images[idx][0], torch.tensor(idx, dtype=torch.int64)
    
class LatentsDataset(torch.utils.data.Dataset):
    def __init__(self, latents_path, flat=False, subset=-1, layer=-1):
        super().__init__()
        self.latents = torch.load(latents_path)
        if flat:
            self.latents = self.latents.view(self.latents.shape[0], -1)
        elif layer == 0:
            self.latents = self.latents[:, 0]
        elif layer > 0:
            self.latents = self.latents[:, layer-1:layer+1]
        if subset > 0:
            self.latents = self.latents[:subset]

        # normalization
        self.mean = self.latents.mean(dim=0).detach()
        self.std = self.latents.std(dim=0).detach()

    @property
    def num_latents(self):
        return self.latents.shape[0]

    @property
    def latent_size(self):
        return self.latents.shape[1:]

    def __len__(self):
        return self.num_latents

    def __getitem__(self, idx):
        latent = self.latents[idx]
        latent = (latent - self.mean) / self.std

        return latent, torch.tensor(idx, dtype=torch.int64)


class ShapeNet(torch.utils.data.Dataset):
    def __init__(self, split='train', sampling=None, root='datasets', 
                 simple_output=False, random_scale=False, subset=-1):
        self.dataset_root = root
        self.sampling = sampling
        self.init_model_bool = False
        self.split = split
        self.simple_output = simple_output
        self.random_scale = random_scale
        self.subset = subset
        self.init_model()
        self.data_type = 'voxel'

    def __len__(self):
        # if self.split == "train":
        #     return 35019
        # else:
        #     return 8762
        return len(self.data_points_int)
    
    def init_model(self):
        split = self.split
        points_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_points_int_' + split + '.pth')
        values_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_values_' + split + '.pth')

        self.data_points_int = torch.load(points_path).byte()
        self.data_values = torch.load(values_path).byte()

        if self.subset > 0:
            self.data_points_int = self.data_points_int[:self.subset]
            self.data_values = self.data_values[:self.subset]

    def __getitem__(self, idx):
        points = (self.data_points_int[idx].float() + 1) / 128 - 1
        # occs = self.data_values[idx].float() * 2 -1
        occs = self.data_values[idx].float()

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        if self.random_scale:
            points = np.random.uniform(0.75, 1.25) * points

        if self.simple_output:
            return occs

        else:
            in_dict = {'idx': idx, 'coords': points}
            gt_dict = {'img': occs}

            return in_dict, gt_dict
        

if __name__ == '__main__':
    lt_path = 'INR_LOE/save/20240503_123320_compute_latents_5_layer_b14_latent_64_30000/latents_train_e_596.pt'
    dataset = LatentsDataset(lt_path, flat=False, layer=4)
