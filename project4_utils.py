import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer

from project4_dataset import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern='', verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split('|')

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f'Разморожен слой: {name}')
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)

        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)

        # Пробовал добавлять массу в модель, но ситуация не улучшилась, видимо где-то ошибка
        self.mass_fc = nn.Sequential(
            nn.Linear(1, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM),
            nn.ReLU(),
        )

        self.calculator = nn.Sequential(
#            nn.Linear(config.HIDDEN_DIM * 3, config.HIDDEN_DIM),
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

#    def forward(self, image, mass, input_ids, attention_mask):
    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)
#        mass_features = self.mass_fc(mass)
        text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]

        image_emb = self.image_proj(image_features)
        text_emb = self.text_proj(text_features)

#        fused_emb = torch.cat((image_emb, mass_features, text_emb), dim=1)
        fused_emb = torch.cat((image_emb, text_emb), dim=1)

        calories = self.calculator(fused_emb).squeeze(dim=1)
        return calories


def train(config, device):
    seed_everything(config.SEED)

    result_epochs = []
    result_train = []
    result_val = []
    worst5 = []

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
#        'params': model.mass_fc.parameters(),
#        'lr': config.MASS_LR
#    }, {
        'params': model.calculator.parameters(),
        'lr': config.CALCULATOR_LR
    }])

    criterion = nn.L1Loss(reduction='none')

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type='test')
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type='test')
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    best_mae = 1e6

    print("training mod started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        result_epochs.append(epoch)

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'image': batch['image'].to(device),
#                'mass': batch['mass'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }
            labels = batch['label'].to(device)

            # Forward
            optimizer.zero_grad()
            calories = model(**inputs)
            loss = criterion(calories, labels).mean()

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Валидация
        val_mae, worst5 = validate(model, val_loader, device, criterion)
        result_val.append(val_mae)

        train_mae = total_loss/len(train_loader)
        result_train.append(train_mae)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train MAE: {train_mae:.2f}| Val MAE: {val_mae:.2f}"
        )

        if val_mae < best_mae:
            print(f"New best model, epoch: {epoch}")
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)

    return result_epochs, result_train, result_val, worst5


def validate(model, val_loader, device, criterion):
    model.eval()

    worst5 = []
    loss_accumulator = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'image': batch['image'].to(device),
#                'mass': batch['mass'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }
            labels = batch['label'].to(device)

            calories = model(**inputs)
            criterion_tensor = criterion(calories, labels)
            loss_accumulator += criterion_tensor.mean().item()

            vals, indices = criterion_tensor.sort(descending=True)
            tmp_worst = [{ 'mae': int(v), 'img': batch['image_path'][indices[i]] } for i, v in enumerate(vals[:5])]
            new_worst = []
            for v in sorted(worst5 + tmp_worst, key=lambda x: x['mae'], reverse=True):
                found = False
                for w in new_worst:
                    if w['img'] == v['img']:
                        found = True
                if not found:
                    new_worst.append(v)
            worst5 = new_worst[:5]

    return loss_accumulator / len(val_loader), worst5
