#!/usr/bin/env python3
"""Cross-continent, source-only crop/weed species recognition on an iNatAg-mini subset.

Run from the project-local environment after inspect_inatag_candidates.py. The held-out
continent is used only for final evaluation; it is not used to tune models or prompts.
"""
from __future__ import annotations
import gc
import json
import math
import os
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

BASE = Path('/home/ubuntu/research')
MANIFEST = BASE / 'inatag_mini_candidate_manifest.csv'
ARCHIVES = BASE / 'archives'
DATA_DIR = BASE / 'selected_images'
OUT = BASE / 'results'
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Two iNatAg-tagged crops and two iNatAg-tagged weeds with substantial NA and Europe counts.
CLASS_ORDER = [
    'zea_mays',                    # crop: maize
    'glycine_max',                 # crop: soybean
    'ambrosia_artemisiifolia',     # weed: common ragweed
    'raphanus_raphanistrum',       # weed: wild radish
]
DISPLAY = {
    'zea_mays': 'Zea mays (maize; crop)',
    'glycine_max': 'Glycine max (soybean; crop)',
    'ambrosia_artemisiifolia': 'Ambrosia artemisiifolia (common ragweed; weed)',
    'raphanus_raphanistrum': 'Raphanus raphanistrum (wild radish; weed)',
}
COMMON = {
    'zea_mays': 'maize or corn',
    'glycine_max': 'soybean',
    'ambrosia_artemisiifolia': 'common ragweed',
    'raphanus_raphanistrum': 'wild radish',
}
SEED = 42
EPOCHS = 30
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 2
DEVICE = torch.device('cpu')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BIO_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIO_STD = (0.26862954, 0.26130258, 0.27577711)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def make_transforms(mean, std):
    train = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.8, 1.25),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ev = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, ev


class ImageRecords(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row['local_path']) as im:
            image = im.convert('RGB')
        return self.transform(image), int(row['label_id'])


def materialize_selected_images(frame):
    """Extract only images in the North America source or Europe target from per-class ZIPs."""
    selected = frame.copy()
    paths = []
    for species, group in selected.groupby('species', sort=False):
        archive = ARCHIVES / f'{species}.zip'
        if not archive.exists():
            raise FileNotFoundError(f'Missing public iNatAg-mini archive: {archive}')
        species_dir = DATA_DIR / species
        species_dir.mkdir(parents=True, exist_ok=True)
        wanted = {str(int(pid)) for pid in group.photo_id}
        found = {}
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.endswith('/') or '/annotations.json' in member:
                    continue
                base = Path(member).name
                stem = Path(base).stem
                if stem in wanted:
                    found[stem] = member
            missing = wanted - set(found)
            if missing:
                raise RuntimeError(f'{species}: {len(missing)} photo IDs missing from archive, e.g. {sorted(missing)[:3]}')
            for pid, member in found.items():
                suffix = Path(member).suffix.lower() or '.jpg'
                target = species_dir / f'{pid}{suffix}'
                if not target.exists():
                    with zf.open(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        path_by_id = {p.stem: str(p) for p in species_dir.iterdir() if p.is_file()}
        paths.extend(path_by_id[str(int(pid))] for pid in group.photo_id)
    selected['local_path'] = paths
    return selected


def prepare_splits():
    if not MANIFEST.exists():
        raise FileNotFoundError(f'Run {BASE / "inspect_inatag_candidates.py"} first.')
    df = pd.read_csv(MANIFEST)
    df = df[df.species.isin(CLASS_ORDER) & df.continent.isin(['North America', 'Europe'])].copy()
    # Require exact data labels and a one-to-one class/status mapping.
    expected_status = {'zea_mays': 'Crop', 'glycine_max': 'Crop',
                       'ambrosia_artemisiifolia': 'Weed', 'raphanus_raphanistrum': 'Weed'}
    for sp, status in expected_status.items():
        observed = set(df.loc[df.species == sp, 'crop_weed'].dropna().str.title())
        if observed != {status}:
            raise ValueError(f'Unexpected iNatAg crop/weed tags for {sp}: {observed}')
    df['label_id'] = df.species.map({s: i for i, s in enumerate(CLASS_ORDER)})
    df = materialize_selected_images(df)
    source = df[df.continent == 'North America'].reset_index(drop=True)
    target = df[df.continent == 'Europe'].reset_index(drop=True)
    # User-grouped source-only validation; no uploader appears in both source train/validation.
    groups = source.user.fillna('').astype(str).to_numpy()
    groups = np.array([g if g else f'photo_{pid}' for g, pid in zip(groups, source.photo_id)])
    y = source.label_id.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(source)), y, groups))
    train = source.iloc[tr_idx].copy()
    val = source.iloc[va_idx].copy()
    if set(train.label_id) != set(range(len(CLASS_ORDER))) or set(val.label_id) != set(range(len(CLASS_ORDER))):
        raise ValueError('The grouped source train/validation fold does not contain every species.')
    if set(train.user.dropna()) & set(val.user.dropna()):
        raise ValueError('Source train/validation uploader leakage detected.')
    if set(source.user.dropna()) & set(target.user.dropna()):
        raise ValueError('Source-target uploader overlap detected.')
    train['split'] = 'source_train'
    val['split'] = 'source_validation'
    target['split'] = 'target_test'
    all_split = pd.concat([train, val, target], ignore_index=True)
    all_split.to_csv(OUT / 'split_manifest.csv', index=False)
    count_table = all_split.groupby(['split', 'species', 'crop_weed']).size().rename('n').reset_index()
    count_table.to_csv(OUT / 'split_counts.csv', index=False)
    print('SPLIT_COUNTS')
    print(count_table.to_string(index=False))
    print('unique_train_users', train.user.nunique(), 'unique_val_users', val.user.nunique(),
          'shared source users', len(set(train.user.dropna()) & set(val.user.dropna())))
    print('unique_source_target_users_shared', len(set(source.user.dropna()) & set(target.user.dropna())))
    return train, val, target


def make_class_weights(train):
    counts = np.bincount(train.label_id.to_numpy(), minlength=len(CLASS_ORDER))
    weights = len(train) / (len(CLASS_ORDER) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def make_loader(frame, transform, shuffle=False):
    ds = ImageRecords(frame, transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)


def build_cnn(name):
    if name == 'ResNet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, len(CLASS_ORDER))
    elif name == 'EfficientNet-B0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_ORDER))
    elif name == 'MobileNetV3-Large':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASS_ORDER))
    else:
        raise ValueError(name)
    return m.to(DEVICE)


def get_logits(model, x):
    return model(x)


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        ys.extend(y.numpy().tolist())
        preds.extend(p.argmax(1).cpu().tolist())
        probs.extend(p.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(preds), np.asarray(probs)


def epoch_metrics(model, loader, criterion):
    y_true, y_pred, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()) * len(y))
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return (sum(losses) / max(1, len(y_true)),
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='macro', zero_division=0))


def train_cnn(name, train, val, class_weights):
    train_tf, eval_tf = make_transforms(IMAGENET_MEAN, IMAGENET_STD)
    train_loader = make_loader(train, train_tf, shuffle=True)
    train_eval_loader = make_loader(train, eval_tf, shuffle=False)
    val_loader = make_loader(val, eval_tf, shuffle=False)
    model = build_cnn(name)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = []
    best_f1 = -1.0
    best_path = OUT / f'{name.replace("-", "_").replace(" ", "_")}_best.pt'
    for epoch in range(1, EPOCHS + 1):
        model.train(); seen = 0; loss_sum = 0.0; correct = 0
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item()); seen += len(y)
        scheduler.step()
        train_loss = loss_sum / max(1, seen)
        train_acc = correct / max(1, seen)
        val_loss, val_acc, val_f1 = epoch_metrics(model, val_loader, criterion)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                        'val_loss': val_loss, 'val_accuracy': val_acc, 'val_macro_f1': val_f1,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
        print(f'{name} epoch {epoch:02d}/{EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}', flush=True)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    pd.DataFrame(history).to_csv(OUT / f'{name.replace("-", "_").replace(" ", "_")}_history.csv', index=False)
    return model, history, best_f1


class BioCLIPLinear(nn.Module):
    def __init__(self, encoder, dim=512):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters(): p.requires_grad = False
        self.encoder.eval()
        self.head = nn.Linear(dim, len(CLASS_ORDER))
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self
    def forward(self, x):
        feats = self.encoder.encode_image(x)
        feats = nn.functional.normalize(feats.float(), dim=-1)
        return self.head(feats)


def load_bioclip():
    import open_clip
    model, _, eval_tf = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    return model.to(DEVICE), tokenizer, eval_tf


@torch.no_grad()
def bioclip_zero_shot(encoder, tokenizer, target):
    prompts = []
    for species in CLASS_ORDER:
        scientific = species.replace('_', ' ')
        role = 'weed' if species in ('ambrosia_artemisiifolia', 'raphanus_raphanistrum') else 'crop'
        names = [
            f'a photo of the plant species {scientific}',
            f'a photo of {COMMON[species]}',
            f'a field photograph of a {COMMON[species]} plant',
            f'a photo of a {role} plant called {COMMON[species]}',
        ]
        tokens = tokenizer(names).to(DEVICE)
        text_features = nn.functional.normalize(encoder.encode_text(tokens).float(), dim=-1)
        prompts.append(nn.functional.normalize(text_features.mean(0), dim=0))
    text_features = torch.stack(prompts)
    _, bio_eval = make_transforms(BIO_MEAN, BIO_STD)
    loader = make_loader(target, bio_eval, shuffle=False)
    ys, preds, probs = [], [], []
    encoder.eval()
    for x, y in loader:
        image_features = nn.functional.normalize(encoder.encode_image(x.to(DEVICE)).float(), dim=-1)
        logits = 100.0 * image_features @ text_features.T
        p = torch.softmax(logits, dim=1)
        ys.extend(y.tolist()); preds.extend(p.argmax(1).cpu().tolist()); probs.extend(p.cpu().numpy().tolist())
    return np.asarray(ys), np.asarray(preds), np.asarray(probs)


def train_bioclip_linear(encoder, train, val, class_weights):
    train_tf, eval_tf = make_transforms(BIO_MEAN, BIO_STD)
    train_loader = make_loader(train, train_tf, shuffle=True)
    val_loader = make_loader(val, eval_tf, shuffle=False)
    model = BioCLIPLinear(encoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = []; best_f1 = -1.0
    best_path = OUT / 'BioCLIP_B16_linear_best.pt'
    for epoch in range(1, EPOCHS + 1):
        model.train(); seen = 0; loss_sum = 0.0; correct = 0
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                feats = nn.functional.normalize(encoder.encode_image(x).float(), dim=-1)
            logits = model.head(feats)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item()); seen += len(y)
        scheduler.step()
        train_loss = loss_sum / max(1, seen); train_acc = correct / max(1, seen)
        val_loss, val_acc, val_f1 = epoch_metrics(model, val_loader, criterion)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                        'val_loss': val_loss, 'val_accuracy': val_acc, 'val_macro_f1': val_f1,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        if val_f1 > best_f1:
            best_f1 = val_f1; torch.save(model.head.state_dict(), best_path)
        print(f'BioCLIP-B/16-linear epoch {epoch:02d}/{EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}', flush=True)
    model.head.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    pd.DataFrame(history).to_csv(OUT / 'BioCLIP_B16_linear_history.csv', index=False)
    return model, history, best_f1


def save_model_report(name, y_true, y_pred):
    labels = list(range(len(CLASS_ORDER)))
    names = [DISPLAY[s] for s in CLASS_ORDER]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   digits=4, zero_division=0)
    (OUT / f'{name.replace("/", "_").replace(" ", "_")}_classification_report.txt').write_text(report + '\n')
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(labels, names, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(labels, names, fontsize=8)
    ax.set_xlabel('Predicted species'); ax.set_ylabel('True species')
    ax.set_title(f'{name}: Europe held-out confusion matrix (n={len(y_true)})')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fname = name.replace('/', '_').replace(' ', '_').replace('-', '_')
    fig.savefig(OUT / f'{fname}_confusion_matrix.png', dpi=180)
    plt.close(fig)
    return report, cm


def metric_row(name, y_true, y_pred, n_params=None, val_f1=None):
    return {'model': name, 'n_target_test': int(len(y_true)),
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'source_validation_macro_f1': val_f1,
            'parameters': n_params}


def plot_learning_curves(histories):
    names = list(histories.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        h = pd.DataFrame(histories[name])
        ax.plot(h.epoch, h.train_accuracy, label='Train', color='#1f77b4')
        ax.plot(h.epoch, h.val_accuracy, label='Source validation', color='#d62728')
        ax.set_title(name); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Source-only training and source validation accuracy (30 epochs)')
    fig.tight_layout()
    fig.savefig(OUT / 'accuracy_curves.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        h = pd.DataFrame(histories[name])
        ax.plot(h.epoch, h.train_loss, label='Train', color='#1f77b4')
        ax.plot(h.epoch, h.val_loss, label='Source validation', color='#d62728')
        ax.set_title(name); ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted cross-entropy loss'); ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Source-only training and source validation loss (30 epochs)')
    fig.tight_layout()
    fig.savefig(OUT / 'loss_curves.png', dpi=180); plt.close(fig)


def normalize_attr(attr):
    a = attr.detach().abs().mean(dim=1, keepdim=True)
    a = a - a.amin(dim=(2, 3), keepdim=True)
    a = a / (a.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return a


def deletion_auc(model, image, attr_map, target_class):
    # Low deletion AUC is preferable: masking top-attribution pixels should rapidly reduce confidence.
    order = torch.argsort(attr_map.flatten(), descending=True)
    h, w = attr_map.shape[-2:]
    total = h * w
    fractions = [0.0, .1, .2, .3, .4, .5]
    scores = []
    model.eval()
    for frac in fractions:
        k = int(frac * total)
        masked = image.clone()
        if k:
            positions = order[:k]
            flat = masked[0].reshape(3, -1)
            flat[:, positions] = 0.0
        with torch.no_grad():
            p = torch.softmax(model(masked), dim=1)[0, target_class].item()
        scores.append(p)
    auc = float(np.trapezoid(scores, fractions) / (fractions[-1] - fractions[0]))
    return auc, scores


def run_xai(model_name, model, target, mean, std):
    from captum.attr import IntegratedGradients, Saliency, Occlusion
    _, eval_tf = make_transforms(mean, std)
    model.eval()
    sample_rows = []
    for label_id in range(len(CLASS_ORDER)):
        candidates = target[target.label_id == label_id].sort_values('photo_id')
        sample_rows.extend([r for _, r in candidates.head(3).iterrows()])
    ig = IntegratedGradients(model)
    sal = Saliency(model)
    occ = Occlusion(model)
    method_aucs = {'Integrated Gradients': [], 'Input Saliency': [], 'Occlusion': []}
    viz_rows = []
    for i, row in enumerate(sample_rows):
        with Image.open(row.local_path) as im:
            original = im.convert('RGB')
            x = eval_tf(original).unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        with torch.no_grad():
            base_logits = model(x)
            pred = int(base_logits.argmax(1).item())
        # Explain the model's own predicted class; accuracy is recorded separately.
        a_ig = ig.attribute(x, baselines=torch.zeros_like(x), target=pred, n_steps=8)
        a_sal = sal.attribute(x, target=pred, abs=True)
        a_occ = occ.attribute(x, baselines=torch.zeros_like(x), target=pred,
                              sliding_window_shapes=(3, 56, 56), strides=(3, 56, 56))
        attrs = {'Integrated Gradients': normalize_attr(a_ig),
                 'Input Saliency': normalize_attr(a_sal),
                 'Occlusion': normalize_attr(a_occ)}
        maps = {k: v[0, 0].detach() for k, v in attrs.items()}
        for method, amap in maps.items():
            auc, curve = deletion_auc(model, x.detach(), amap, pred)
            method_aucs[method].append(auc)
        sample_rows[i] = {**row.to_dict(), 'predicted': pred, 'correct': pred == int(row.label_id),
                          'confidence': float(torch.softmax(base_logits,1)[0,pred].item()),
                          'image': original.copy(), 'maps': maps}
    # Quantitative method-level faithfulness summary on a stratified 12-image Europe sample.
    summary = pd.DataFrame([{'method': k, 'n_explanations': len(v),
                             'mean_deletion_auc_predicted_confidence': float(np.mean(v)),
                             'std_deletion_auc': float(np.std(v, ddof=1)) if len(v)>1 else 0.0}
                            for k,v in method_aucs.items()])
    summary['interpretation'] = 'Lower deletion AUC indicates faster confidence reduction when top-attribution pixels are removed.'
    summary.to_csv(OUT / 'xai_deletion_summary.csv', index=False)
    pd.DataFrame([{k: r[k] for k in ['photo_id','species','continent','crop_weed','label_id','predicted','correct','confidence']}
                  for r in sample_rows]).to_csv(OUT / 'xai_sample_predictions.csv', index=False)
    # One target-domain image per class, same original for each explanation method.
    fig, axes = plt.subplots(len(CLASS_ORDER), 4, figsize=(14, 3.4 * len(CLASS_ORDER)))
    colnames = ['Original', 'Integrated Gradients', 'Input Saliency', 'Occlusion']
    for row_idx, label_id in enumerate(range(len(CLASS_ORDER))):
        sample = next(r for r in sample_rows if int(r['label_id']) == label_id)
        im = sample['image'].resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        axes[row_idx, 0].imshow(im); axes[row_idx, 0].set_title(f"{DISPLAY[CLASS_ORDER[label_id]]}\nPred: {DISPLAY[CLASS_ORDER[sample['predicted']]]} ({'correct' if sample['correct'] else 'error'})", fontsize=8)
        for j, method in enumerate(colnames[1:], start=1):
            heat = sample['maps'][method].cpu().numpy()
            axes[row_idx, j].imshow(im)
            axes[row_idx, j].imshow(heat, cmap='jet', alpha=.42, vmin=0, vmax=1)
            axes[row_idx, j].set_title(method, fontsize=9)
        for ax in axes[row_idx]: ax.axis('off')
    fig.suptitle(f'XAI on held-out Europe images — source-validation winner: {model_name}\nQualitative maps explain the model prediction; deletion AUC is summarized over 12 images.', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'xai_comparison.png', dpi=180); plt.close(fig)
    return summary


def main():
    set_seed()
    print('hardware', {'device': str(DEVICE), 'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'cpu_threads': torch.get_num_threads()})
    train, val, target = prepare_splits()
    class_weights = make_class_weights(train)
    histories = {}; models_by_name = {}; val_f1s = {}; param_counts = {}
    for name in ['ResNet18', 'EfficientNet-B0', 'MobileNetV3-Large']:
        model, hist, best_f1 = train_cnn(name, train, val, class_weights)
        histories[name] = hist; models_by_name[name] = model; val_f1s[name] = best_f1
        param_counts[name] = sum(p.numel() for p in model.parameters())
        gc.collect()

    bio_encoder, tokenizer, _ = load_bioclip()
    # Zero-shot BioCLIP uses no iNatAg source labels or target labels for fitting.
    y_bio0, pred_bio0, prob_bio0 = bioclip_zero_shot(bio_encoder, tokenizer, target)
    bio_linear, hist, best_f1 = train_bioclip_linear(bio_encoder, train, val, class_weights)
    histories['BioCLIP-B/16 linear probe'] = hist
    models_by_name['BioCLIP-B/16 linear probe'] = bio_linear
    val_f1s['BioCLIP-B/16 linear probe'] = best_f1
    param_counts['BioCLIP-B/16 linear probe'] = sum(p.numel() for p in bio_linear.parameters())
    del tokenizer
    gc.collect()

    test_loader_cnn = {name: make_loader(target, make_transforms(IMAGENET_MEAN, IMAGENET_STD)[1], shuffle=False)
                       for name in ['ResNet18','EfficientNet-B0','MobileNetV3-Large']}
    y_test = target.label_id.to_numpy()
    metrics = []
    reports = {}
    for name in ['ResNet18','EfficientNet-B0','MobileNetV3-Large']:
        ys, pred, probs = predict_loader(models_by_name[name], test_loader_cnn[name])
        reports[name], _ = save_model_report(name, ys, pred)
        metrics.append(metric_row(name, ys, pred, param_counts[name], val_f1s[name]))
        pd.DataFrame({'photo_id':target.photo_id, 'true_label':ys, 'predicted_label':pred,
                      'true_species':[CLASS_ORDER[i] for i in ys], 'predicted_species':[CLASS_ORDER[i] for i in pred],
                      'max_probability':probs.max(1)}).to_csv(OUT / f'{name.replace("-","_")}_target_predictions.csv', index=False)
    _, bio_eval = make_transforms(BIO_MEAN, BIO_STD)
    bio_loader = make_loader(target, bio_eval, shuffle=False)
    ys, pred, probs = predict_loader(models_by_name['BioCLIP-B/16 linear probe'], bio_loader)
    reports['BioCLIP-B/16 linear probe'], _ = save_model_report('BioCLIP-B/16 linear probe', ys, pred)
    metrics.append(metric_row('BioCLIP-B/16 linear probe', ys, pred, param_counts['BioCLIP-B/16 linear probe'], val_f1s['BioCLIP-B/16 linear probe']))
    reports['BioCLIP zero-shot'], _ = save_model_report('BioCLIP zero-shot', y_bio0, pred_bio0)
    metrics.append(metric_row('BioCLIP zero-shot', y_bio0, pred_bio0, None, None))
    pd.DataFrame(metrics).to_csv(OUT / 'model_metrics.csv', index=False)
    (OUT / 'classification_reports.json').write_text(json.dumps(reports, indent=2))
    plot_learning_curves(histories)

    # Choose the winner only on the North American, uploader-grouped source validation fold.
    winner = max(val_f1s, key=val_f1s.get)
    print('SOURCE_VALIDATION_WINNER', winner, 'macro_f1', val_f1s[winner], flush=True)
    if winner.startswith('BioCLIP'):
        mean, std = BIO_MEAN, BIO_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    xai_summary = run_xai(winner, models_by_name[winner], target, mean, std)
    print('XAI_DELETION', xai_summary.to_dict(orient='records'))

    config = {
        'dataset': 'Project-AgML/iNatAg-mini', 'full_iNatAg_images': 4720903,
        'full_iNatAg_species': 2959, 'mini_images': 560844, 'mini_sampling': 'up to 200 per species',
        'selected_species_order': CLASS_ORDER, 'source_continent': 'North America', 'target_continent': 'Europe',
        'zero_target_label_training': True,
        'meaning_of_zero_shot': 'zero target-continent examples/labels during model fitting; the four species classes are known from source training',
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'image_size': IMAGE_SIZE,
        'augmentations': ['random resized crop scale .75-1.0', 'horizontal/vertical flips p=.5', 'rotation +/-20 degrees', 'color jitter brightness/contrast .18, saturation .15, hue .03'],
        'augmentation_policy': 'on-the-fly, one random view per training image per epoch; no offline image copies; no validation/test augmentation',
        'cnn_optimizer': 'AdamW lr 1e-4, weight decay 1e-4, cosine annealing 30 epochs, weighted cross-entropy',
        'bioclip_linear_optimizer': 'AdamW classifier-head lr 1e-3, weight decay 1e-4, cosine annealing 30 epochs, frozen BioCLIP visual encoder, weighted cross-entropy',
        'validation': 'one fold of StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42) on source only, grouped by uploader',
        'best_checkpoint': 'highest source-validation macro-F1; target metrics are not used to choose winner',
        'seed': SEED, 'hardware': 'CPU-only ARM64 sandbox, PyTorch CPU; 6 reported CPUs; no CUDA',
        'class_display': DISPLAY,
        'augmentation_counting': 'Unique image count is unchanged by online transforms. Each model sees 30 x N_train stochastic views over 30 epochs.'
    }
    (OUT / 'run_config.json').write_text(json.dumps(config, indent=2))
    summary = {
        'n_source_train': len(train), 'n_source_validation': len(val), 'n_target_europe_test': len(target),
        'n_unique_selected_source_and_target': len(train)+len(val)+len(target),
        'n_source_train_views_per_model_30_epochs': len(train)*EPOCHS,
        'epochs': EPOCHS, 'winner_by_source_validation': winner,
        'winner_source_validation_macro_f1': val_f1s[winner],
        'target_europe_metrics': metrics,
    }
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print('RUN_SUMMARY', json.dumps(summary, indent=2), flush=True)
    print('OUTPUT_DIR', OUT, flush=True)

if __name__ == '__main__':
    main()
