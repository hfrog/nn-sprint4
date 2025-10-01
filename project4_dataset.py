import os
import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

import albumentations as A


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type='train'):
        self.config = config
        self.transforms = transforms

        self.df = pd.read_csv(config.DISH_DF_PATH)
        self.df = self.df[self.df.split == ds_type]

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # dish_id            object
        # total_calories    float64
        # total_mass        float64
        # ingredients        object

        df_item = self.df.loc[self.df.index[idx]]

        img_path = os.path.join(self.config.IMAGES_PATH, df_item.dish_id, 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image=np.array(image))['image']

        label = df_item.total_calories
        mass = df_item.total_mass
        ingredients = df_item.ingredients

        return {'image': image, 'image_path': img_path, 'mass': mass, 'ingredients': ingredients, 'label': label}


def collate_fn(batch, tokenizer):
    images = torch.stack([item['image'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    masses = torch.tensor([item['mass'] for item in batch], dtype=torch.float).unsqueeze(1)
    ingredients = [item['ingredients'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

    tokenized_input = tokenizer(ingredients,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True)

    return {
        'image': images,
        'image_path': image_paths,
        'mass': masses,
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'label': labels,
    }


def get_transforms(config, ds_type='train'):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == 'train':
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
#                A.RandomCrop(
#                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
#                A.CoarseDropout(
#                    num_holes_range=(2, 8),
#                    hole_height_range=(int(0.07 * cfg.input_size[1]),
#                                       int(0.15 * cfg.input_size[1])),
#                    hole_width_range=(int(0.1 * cfg.input_size[2]),
#                                      int(0.15 * cfg.input_size[2])),
#                    fill=0,
#                    p=0.5),
#                A.ColorJitter(brightness=0.2,
#                              contrast=0.2,
#                              saturation=0.2,
#                              hue=0.1,
#                              p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=42,
        )

    return transforms
