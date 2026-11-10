#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, '/home/ubuntu/research')
import train_inatag_crosscontinent as t

def main():
    t.set_seed()
    train, val, target = t.prepare_splits()
    encoder, tokenizer, _ = t.load_bioclip()
    del tokenizer
    model = t.BioCLIPLinear(encoder).to(t.DEVICE)
    path = t.OUT / 'BioCLIP_B16_linear_best.pt'
    model.head.load_state_dict(t.torch.load(path, map_location=t.DEVICE, weights_only=True))
    model.eval()
    summary = t.run_xai('BioCLIP-B/16 linear probe', model, target, t.BIO_MEAN, t.BIO_STD)
    print(summary.to_string(index=False))
    # Regenerate config and summary metadata that the original run would have written after XAI.
    metrics = t.pd.read_csv(t.OUT / 'model_metrics.csv').to_dict(orient='records')
    config = {
        'dataset':'Project-AgML/iNatAg-mini', 'full_iNatAg_images':4720903,
        'full_iNatAg_species':2959, 'mini_images':560844, 'mini_sampling':'up to 200 per species',
        'selected_species_order':t.CLASS_ORDER, 'source_continent':'North America','target_continent':'Europe',
        'zero_target_label_training':True,
        'meaning_of_zero_shot':'zero target-continent examples/labels during model fitting; the four species classes are known from source training',
        'epochs':t.EPOCHS,'batch_size':t.BATCH_SIZE,'image_size':t.IMAGE_SIZE,
        'augmentations':['random resized crop scale .75-1.0','horizontal/vertical flips p=.5','rotation +/-20 degrees','color jitter brightness/contrast .18, saturation .15, hue .03'],
        'augmentation_policy':'on-the-fly, one random view per training image per epoch; no offline image copies; no validation/test augmentation',
        'cnn_optimizer':'AdamW lr 1e-4, weight decay 1e-4, cosine annealing 30 epochs, weighted cross-entropy',
        'bioclip_linear_optimizer':'AdamW classifier-head lr 1e-3, weight decay 1e-4, cosine annealing 30 epochs, frozen BioCLIP visual encoder, weighted cross-entropy',
        'validation':'one fold of StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42) on source only, grouped by uploader',
        'best_checkpoint':'highest source-validation macro-F1; target metrics are not used to choose winner',
        'seed':t.SEED,'hardware':'CPU-only ARM64 sandbox, PyTorch CPU; 6 reported CPUs; no CUDA',
        'class_display':t.DISPLAY,
        'augmentation_counting':'Unique image count is unchanged by online transforms. Each model sees 30 x N_train stochastic views over 30 epochs.'
    }
    (t.OUT/'run_config.json').write_text(t.json.dumps(config,indent=2))
    winner='BioCLIP-B/16 linear probe'
    summary_json={'n_source_train':len(train),'n_source_validation':len(val),'n_target_europe_test':len(target),
        'n_unique_selected_source_and_target':len(train)+len(val)+len(target),
        'n_source_train_views_per_model_30_epochs':len(train)*t.EPOCHS,'epochs':t.EPOCHS,
        'winner_by_source_validation':winner,'winner_source_validation_macro_f1':0.9722954179475919,
        'target_europe_metrics':metrics}
    (t.OUT/'run_summary.json').write_text(t.json.dumps(summary_json,indent=2))
    print('Wrote run_config.json and run_summary.json')

if __name__=='__main__': main()
