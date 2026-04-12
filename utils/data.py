
# Code adapted from:
# https://github.com/yandex-research/switti/blob/master/utils/data.py

import csv
import os
import random

import PIL.Image as PImage
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def normalize_255_into_pm1(x):
    return x.div_(127.5).sub(-1)


class COCODataset(Dataset):
    def __init__(self, root_dir, subset_name="subset", transform=None, max_cnt=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (
            "jpg",
            "jpeg",
            "png",
            "ppm",
            "bmp",
            "pgm",
            "tif",
            "tiff",
            "webp",
        )
        sample_dir = os.path.join(root_dir, subset_name)

        # Collect sample paths
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.split(".")[-1] in self.extensions
            ],
            key=lambda x: x.split("/")[-1].split(".")[0],
        )
        # restrict num samples
        self.samples = self.samples if max_cnt is None else self.samples[:max_cnt]

        # Collect captions
        self.captions = {}
        with open(os.path.join(root_dir, f"{subset_name}.csv"), newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(spamreader):
                if i == 0:
                    continue
                self.captions[row[1]] = row[2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        return sample, self.captions[os.path.basename(sample_path)]


def coco_collate_fn(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        max_cnt=None,
        repeat=10,
        placeholder_token="<sks>",
        learnable_property="object",
        use_captions=True,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (
            "jpg",
            "jpeg",
            "png",
            "ppm",
            "bmp",
            "pgm",
            "tif",
            "tiff",
            "webp",
        )
        sample_dir = os.path.join(root_dir, "imgs")
        self.num_samples = len(os.listdir(sample_dir))
        self._length = self.num_samples * repeat

        # Collect sample paths
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.split(".")[-1] in self.extensions
            ],
            key=lambda x: x.split("/")[-1].split(".")[0],
        )
        # restrict num samples
        self.samples = self.samples if max_cnt is None else self.samples[:max_cnt]

        # Collect captions
        self.captions = {}

        if use_captions and os.path.exists(os.path.join(root_dir, "captions.csv")):
            with open(os.path.join(root_dir, "captions.csv"), newline="\n") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=",")
                for i, row in enumerate(spamreader):
                    if i == 0:
                        continue
                    self.captions[row[1]] = row[2]

        # print("Use caption", use_captions)
        print(self.captions)

        # placeholder token
        self.placeholder_token = placeholder_token

        # template
        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx % self.num_samples

        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        caption = random.choice(self.templates).format(self.placeholder_token)
        if self.captions:
            caption = self.captions[os.path.basename(sample_path)].format(self.placeholder_token)

        # print(caption)
        return sample, caption


class TextualInversionDatasetDisentanglement(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        max_cnt=None,
        repeat=10,
        placeholder_token_content="<object>",
        placeholder_token_style="<style>",
        use_captions=True,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = (
            "jpg",
            "jpeg",
            "png",
            "ppm",
            "bmp",
            "pgm",
            "tif",
            "tiff",
            "webp",
        )
        sample_dir = os.path.join(root_dir, "imgs")
        self.num_samples = len(os.listdir(sample_dir))
        self._length = self.num_samples * repeat

        # Collect sample paths
        self.samples = sorted(
            [
                os.path.join(sample_dir, fname)
                for fname in os.listdir(sample_dir)
                if fname.split(".")[-1] in self.extensions
            ],
            key=lambda x: x.split("/")[-1].split(".")[0],
        )
        # restrict num samples
        self.samples = self.samples if max_cnt is None else self.samples[:max_cnt]

        # Collect captions
        self.use_captions = use_captions
        self.captions_full = {}
        self.captions_object = {}
        self.captions_style = {}

        if self.use_captions and os.path.exists(os.path.join(root_dir, "captions.csv")):
            with open(os.path.join(root_dir, "captions.csv"), newline="\n") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=",")
                for i, row in enumerate(spamreader):
                    if i == 0:
                        continue
                    self.captions_full[row[1]] = row[2]
                    self.captions_object[row[1]] = row[3]
                    self.captions_style[row[1]] = row[4]

        # print("Use caption", use_captions)
        print(self.captions_full)

        # placeholder token
        self.placeholder_token_content = placeholder_token_content
        self.placeholder_token_style = placeholder_token_style

        # template
        self.content_templates = imagenet_templates_small
        self.style_templates = imagenet_style_templates_small

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx % self.num_samples

        sample_path = self.samples[idx]
        sample = Image.open(sample_path).convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        caption_full = f"A {self.placeholder_token_content} in {self.placeholder_token_style}"
        caption_content = random.choice(self.content_templates).format(
            self.placeholder_token_content
        )
        caption_style = random.choice(self.style_templates).format(self.placeholder_token_style)
        if self.use_captions:
            caption_full = self.captions_full[os.path.basename(sample_path)].format(
                self.placeholder_token_content, self.placeholder_token_style
            )
            caption_content = self.captions_object[os.path.basename(sample_path)].format(
                self.placeholder_token_content
            )
            caption_style = self.captions_style[os.path.basename(sample_path)].format(
                self.placeholder_token_style
            )

        # caption_content = f'A {self.placeholder_token_content}'
        # caption_style = f'A {self.placeholder_token_style}'
        # caption_content_style = f'A {self.placeholder_token_content} in style of {self.placeholder_token_style}'

        # print(caption_content, caption_style, caption_content_style)
        return sample, caption_content, caption_style, caption_full


def build_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
):
    # build augmentations
    # first resize to mid_reso, then crop to final_reso
    mid_reso = round(mid_reso * final_reso)
    train_aug = [
        transforms.Resize(
            mid_reso,
            interpolation=InterpolationMode.LANCZOS,
        ),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    if hflip:
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug = transforms.Compose(train_aug)

    # build dataset
    train_set = COCODataset(
        data_path,
        subset_name="train2014",
        transform=train_aug,
        max_cnt=None,
    )
    print(f"[Dataset] {len(train_set)=}")
    print_aug(train_aug, "[train]")

    return train_set


def build_personalized_dataset(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
    repeat=10,
    placeholder_token="<sks>",
    learnable_property="object",
    use_captions=True,
):
    # build augmentations
    # first resize to mid_reso, then crop to final_reso
    mid_reso = round(mid_reso * final_reso)
    # train_aug = [
    #     transforms.Resize(
    #         mid_reso,
    #         interpolation=InterpolationMode.LANCZOS,
    #     ),
    #     transforms.CenterCrop((final_reso, final_reso)),
    #     transforms.ToTensor(),
    #     normalize_01_into_pm1,
    # ]
    # if hflip:
    #     train_aug.insert(0, transforms.RandomHorizontalFlip())

    train_aug = [
        transforms.Resize(
            512,
            interpolation=InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    train_aug.insert(0, transforms.RandomHorizontalFlip())

    train_aug = transforms.Compose(train_aug)

    # build dataset
    train_set = TextualInversionDataset(
        data_path,
        transform=train_aug,
        max_cnt=None,
        repeat=repeat,
        placeholder_token=placeholder_token,
        learnable_property=learnable_property,
        use_captions=use_captions,
    )
    print(f"[Dataset] {len(train_set)=}")
    print_aug(train_aug, "[train]")

    return train_set


def build_personalized_dataset_disentanglement(
    data_path: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
    repeat=10,
    placeholder_token_content="<object>",
    placeholder_token_style="<style>",
    use_captions=True,
):
    # build augmentations
    # first resize to mid_reso, then crop to final_reso
    mid_reso = round(mid_reso * final_reso)
    # train_aug = [
    #     transforms.Resize(
    #         mid_reso,
    #         interpolation=InterpolationMode.LANCZOS,
    #     ),
    #     transforms.CenterCrop((final_reso, final_reso)),
    #     transforms.ToTensor(),
    #     normalize_01_into_pm1,
    # ]
    # if hflip:
    #     train_aug.insert(0, transforms.RandomHorizontalFlip())

    train_aug = [
        transforms.Resize(
            512,
            interpolation=InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    train_aug.insert(0, transforms.RandomHorizontalFlip())

    train_aug = transforms.Compose(train_aug)

    # build dataset
    train_set = TextualInversionDatasetDisentanglement(
        data_path,
        transform=train_aug,
        max_cnt=None,
        repeat=repeat,
        placeholder_token_content=placeholder_token_content,
        placeholder_token_style=placeholder_token_style,
        use_captions=use_captions,
    )
    print(f"[Dataset] {len(train_set)=}")
    print_aug(train_aug, "[train]")

    return train_set


def pil_loader(path):
    with open(path, "rb") as f:
        img: PImage.Image = PImage.open(f).convert("RGB")
    return img


def print_aug(transform, label):
    print(f"Transform {label} = ")
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print("---------------------------\n")
