#!/usr/bin/env python3
"""Create separate accuracy and loss plots from the recorded 30-epoch histories."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS=Path('/home/ubuntu/research/results')
OUT=RESULTS/'individual_curves'; OUT.mkdir(parents=True,exist_ok=True)
MODELS=[('ResNet18','ResNet18'),('EfficientNet-B0','EfficientNet_B0'),
        ('MobileNetV3-Large','MobileNetV3_Large'),('BioCLIP ViT-B/16 linear probe','BioCLIP_B16_linear')]
for title,stem in MODELS:
    df=pd.read_csv(RESULTS/f'{stem}_history.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.0))
    ax.plot(df.epoch,df.train_accuracy,color='#1f77b4',lw=2,label='Training accuracy')
    ax.plot(df.epoch,df.val_accuracy,color='#d62728',lw=2,label='Source-validation accuracy')
    ax.set(title=f'{title}: accuracy by epoch',xlabel='Epoch',ylabel='Accuracy',xlim=(1,30),ylim=(0,1.02))
    ax.set_xticks([1,5,10,15,20,25,30]); ax.grid(alpha=.25); ax.legend(loc='lower right',frameon=False)
    fig.tight_layout(); fig.savefig(OUT/f'{stem}_accuracy.png',dpi=200,bbox_inches='tight'); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.0))
    ax.plot(df.epoch,df.train_loss,color='#1f77b4',lw=2,label='Training loss')
    ax.plot(df.epoch,df.val_loss,color='#d62728',lw=2,label='Source-validation loss')
    ax.set(title=f'{title}: weighted cross-entropy loss by epoch',xlabel='Epoch',ylabel='Weighted cross-entropy loss',xlim=(1,30))
    ax.set_xticks([1,5,10,15,20,25,30]); ax.grid(alpha=.25); ax.legend(loc='upper right',frameon=False)
    fig.tight_layout(); fig.savefig(OUT/f'{stem}_loss.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print('Created eight separate accuracy/loss plots in',OUT)
