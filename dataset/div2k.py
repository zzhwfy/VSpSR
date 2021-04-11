from PIL import Image
import random
import os
from torch.utils.data import Dataset
from torchvision import transforms


class Div2kSR(Dataset):
    def __init__(self, lr_folder, hr_folder, crop_size=32, scale=4, agument=True):
        self.crop_size = crop_size
        self.scale = scale
        self.agument = agument
        lr_filenames = sorted(os.listdir(lr_folder))
        hr_filenames = sorted(os.listdir(hr_folder))
        self.files = []

        for lr_filename, hr_filename in zip(lr_filenames, hr_filenames):
            lr_file = os.path.join(lr_folder, lr_filename)
            hr_file = os.path.join(hr_folder, hr_filename)

            self.files.append((lr_file, hr_file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lr_file, hr_file = self.files[idx % len(self.files)]

        img_lr = transforms.ToTensor()(Image.open(lr_file).convert('RGB'))
        img_hr = transforms.ToTensor()(Image.open(hr_file).convert('RGB'))
        # transform

        w_lr = self.crop_size
        x0 = random.randint(0, img_lr.shape[-2] - w_lr)
        y0 = random.randint(0, img_lr.shape[-1] - w_lr)
        crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
        w_hr = w_lr * self.scale
        x1 = x0 * self.scale
        y1 = y0 * self.scale
        crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.agument:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        return crop_lr, crop_hr


def build(image_set, args):
    PATHS = {
        "train": {'HR': 'DIV2K-tr_1X', 4: 'DIV2K-tr_4X', 8: 'DIV2K-tr_8X'},
        "val": {'HR': 'DIV2K-va_1X', 4: 'DIV2K-va_4X', 8: 'DIV2K-va_8X'},
    }

    hr_folder = os.path.join(args.dataset_root, PATHS[image_set]['HR'])
    lr_folder = os.path.join(args.dataset_root, PATHS[image_set][args.scale])
    dataset = Div2kSR(lr_folder, hr_folder, args.crop_size, args.scale, agument=True)

    return dataset
