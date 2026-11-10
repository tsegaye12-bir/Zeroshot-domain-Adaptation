#!/usr/bin/env python3
"""Rebuild manuscript Results in consistent model-by-model order from real outputs."""
from pathlib import Path
import json, re
import pandas as pd

BASE=Path('/home/ubuntu/research'); OUT=BASE/'results'
ARTICLE=BASE/'cross_continental_zero_shot_article.md'
metrics=pd.read_csv(OUT/'model_metrics.csv')
reports=json.loads((OUT/'classification_reports.json').read_text())
order=['Zea mays (maize; crop)','Glycine max (soybean; crop)',
       'Ambrosia artemisiifolia (common ragweed; weed)',
       'Raphanus raphanistrum (wild radish; weed)']
MODELS=[
 ('ResNet18','ResNet18','ResNet18','ResNet18_confusion_matrix.png',
  'ResNet18 correctly classifies 44/51 maize and 50/70 ragweed images. Maize recall is 0.8627 but precision is only 0.6567; the matrix shows 11 ragweed and 8 soybean images predicted as maize. At epoch 30, training accuracy is 1.0000 while source-validation accuracy is 0.8551, and training/validation loss is 0.0360/0.3167. This late-epoch gap is consistent with overfitting risk on a small source sample; it does not by itself identify the cause of errors on Europe.'),
 ('EfficientNet-B0','EfficientNet_B0','EfficientNet-B0','EfficientNet_B0_confusion_matrix.png',
  'EfficientNet-B0 correctly classifies 44/51 maize, 43/53 soybean, 50/70 ragweed, and 48/60 wild radish images. The most prominent error is 19 ragweed images predicted as maize; this coincides with ragweed recall of 0.7143 and maize precision of 0.6286. Its epoch-30 training/validation accuracy is 0.9818/0.8551 and loss is 0.0842/0.4339, while the best source-validation macro-F1 checkpoint occurred earlier at epoch 16. The divergence suggests overfitting risk and motivates using the pre-specified validation-selected checkpoint rather than the last epoch.'),
 ('MobileNetV3-Large','MobileNetV3_Large','MobileNetV3-Large','MobileNetV3_Large_confusion_matrix.png',
  'MobileNetV3-Large is the strongest CNN by Europe accuracy (81.20%) and macro-F1 (81.22%), correctly classifying 190/234 images. It recognizes 46/51 maize, 44/53 soybean, 48/70 ragweed, and 52/60 wild radish. Ragweed recall is lowest at 0.6857, with 13 ragweed images predicted as maize. The selected source-validation checkpoint is epoch 23 (macro-F1 0.9457); the 30-epoch curves show high source fit, but the lower European score demonstrates that source-validation performance does not fully predict this geographic holdout.'),
 ('BioCLIP-B/16 linear probe','BioCLIP_B16_linear','BioCLIP ViT-B/16 linear probe','BioCLIP_B_16_linear_probe_confusion_matrix.png',
  'The source-trained linear probe correctly classifies 220/234 European images and reaches 94.02% accuracy and 94.03% macro-F1. It correctly identifies all 51 maize images; its largest errors are five ragweed and four soybean images predicted as maize. The best source-validation checkpoint occurs at epoch 9 (macro-F1 0.9723); at epoch 30 the frozen-encoder probe remains closely aligned on source training and validation (accuracy 0.9745/0.9565; loss 0.7209/0.7033). This is the strongest result in this pilot, but it does not rule out pretraining overlap or establish performance across other regions and taxa.'),
]

def row(key):
    return metrics.loc[metrics.model==key].iloc[0]
def percent(x): return f'{100*float(x):.2f}%'
def per_class(key):
    found={}
    for line in reports[key].splitlines():
        s=line.strip()
        for cls in order:
            if s.startswith(cls):
                vals=s[len(cls):].split()
                if len(vals)>=4: found[cls]=vals[-4:]
    missing=set(order)-set(found)
    if missing: raise ValueError(f'Missing report rows for {key}: {missing}')
    return found
def report_table(num, key, label):
    rows=['| Taxon | Precision | Recall | F1-score | Support |','|---|---:|---:|---:|---:|']
    for cls in order:
        p,r,f,s=per_class(key)[cls]
        rows.append(f'| {cls} | {p} | {r} | {f} | {s} |')
    averages={}
    for line in reports[key].splitlines():
        s=line.strip()
        for name in ('macro avg','weighted avg'):
            if s.startswith(name):
                vals=s[len(name):].split()
                if len(vals)>=4: averages[name]=vals[-4:]
    for name in ('macro avg','weighted avg'):
        if name not in averages: raise ValueError(f'Missing {name} row in {key} report')
        p,r,f,s=averages[name]
        rows.append(f'| {name.title()} | {p} | {r} | {f} | {s} |')
    return f'**Table {num}. {label}: European holdout classification report (n=234).**\n\n'+'\n'.join(rows)
def figure(n, caption, alt, path):
    full=OUT/path
    if not full.is_file(): raise FileNotFoundError(full)
    return f'**Figure {n}. {caption}**\n\n![{alt}]({full})'
def cm_paragraph(matrix, note):
    return f'**Confusion matrix.** {note} Rows denote true taxa and columns predicted taxa; cells are raw image counts.\n\n{figure(matrix[0],matrix[1],matrix[2],matrix[3])}'

parts=['## 3. Results and Analysis',
       'All four supervised/source-trained configurations completed 30 epochs. The plots below are separate per model and show only source-training and source-validation metrics; the European target set is not plotted or used in checkpoint selection. Each trained model is presented in the same order: accuracy curve, loss curve, European confusion matrix, per-class report, and interpretation. BioCLIP prompt-only is inference without fitting and therefore has no training accuracy or loss curve.']
fig_no=1; table_no=3
for i,(key,stem,title,cmfile,interpretation) in enumerate(MODELS, start=1):
    r=row(key); h=pd.read_csv(OUT/f'{stem}_history.csv')
    best=h.loc[h.val_macro_f1.idxmax()]; last=h.iloc[-1]
    parts += [f'### 3.{i} {title}',
      f'{title} completed 30 epochs. The checkpoint selected by source-validation macro-F1 is epoch {int(best.epoch)} (source-validation accuracy {best.val_accuracy:.4f}; macro-F1 {best.val_macro_f1:.4f}). On the untouched Europe holdout it achieved {percent(r.accuracy)} accuracy, {percent(r.balanced_accuracy)} balanced accuracy, and {percent(r.macro_f1)} macro-F1. The target metrics are reported after checkpoint selection and did not determine it.',
      figure(fig_no,f'{title} source-training and source-validation accuracy over 30 epochs.',f'{title} separate accuracy plot','individual_curves/'+stem+'_accuracy.png')]
    fig_no+=1
    parts.append(f'**Accuracy-curve reading.** At epoch 30 the training accuracy was {last.train_accuracy:.4f} and source-validation accuracy was {last.val_accuracy:.4f}. The best-validation epoch is identified above; target performance is not represented in this curve.')
    parts.append(figure(fig_no,f'{title} source-training and source-validation weighted cross-entropy loss over 30 epochs.',f'{title} separate loss plot','individual_curves/'+stem+'_loss.png'))
    fig_no+=1
    parts.append(f'**Loss-curve reading.** At epoch 30 training loss was {last.train_loss:.4f} and source-validation loss was {last.val_loss:.4f}. The curves diagnose fitting on the source split only; the European holdout remains independent.')
    if key=='ResNet18':
        note='ResNet18 has 182 correct predictions on the diagonal; 11 common-ragweed and 8 soybean observations are assigned to maize.'
    elif key=='EfficientNet-B0':
        note='The largest off-diagonal cell is 19 common-ragweed observations assigned to maize; 48 wild-radish observations are correctly classified.'
    elif key=='MobileNetV3-Large':
        note='The largest off-diagonal cell is 13 common-ragweed observations assigned to maize; wild radish has 52 correct predictions.'
    else:
        note='There are 220 correct predictions on the diagonal; all 51 maize observations are correct, while the largest off-diagonal cells are five ragweed and four soybean observations assigned to maize.'
    parts.append(cm_paragraph((fig_no,f'{title} European holdout confusion matrix (n=234).',f'{title} raw confusion matrix',cmfile),note)); fig_no+=1
    parts.append(report_table(table_no,key,title)); table_no+=1
    parts.append(f'**Interpretation.** {interpretation}')

# Prompt-only has no fit or epoch history, but receives the same target confusion/report analysis.
key='BioCLIP zero-shot'; title='BioCLIP ViT-B/16 prompt-only'; r=row(key)
parts += ['### 3.5 BioCLIP ViT-B/16 prompt-only',
 f'This condition encodes scientific/common-name prompts and compares them with European image embeddings. It uses no iNatAg training labels, no target labels, and no fitted task-specific head; accordingly, training-accuracy and training-loss curves do not exist and are not fabricated. It obtains {percent(r.accuracy)} accuracy, {percent(r.balanced_accuracy)} balanced accuracy, and {percent(r.macro_f1)} macro-F1 on the same 234-image Europe holdout.',
 '**Confusion matrix.** The prompt-only condition correctly classifies 215/234 images. The largest off-diagonal cell is six common-ragweed images assigned to soybean; rows are true taxa and columns are predictions.',
 figure(fig_no,'BioCLIP ViT-B/16 prompt-only European holdout confusion matrix (n=234).','BioCLIP prompt-only raw confusion matrix','BioCLIP_zero_shot_confusion_matrix.png')]
fig_no+=1
parts.append(report_table(table_no,key,title)); table_no+=1
parts.append('**Interpretation.** The high prompt-only score is a genuinely label-free reference for the specified four known classes, not evidence of open-set recognition. Relative to the source-trained BioCLIP probe, its accuracy is lower by 2.14 percentage points; the detailed per-class table shows the remaining class-specific trade-offs.')

# Explainability block precedes the final overall comparative analysis.
xai=pd.read_csv(OUT/'xai_deletion_summary.csv')
xai_rows=['| Explanation method | n | Mean deletion AUC | SD |','|---|---:|---:|---:|']
for _,z in xai.iterrows(): xai_rows.append(f"| {z['method']} | {int(z['n_explanations'])} | {z['mean_deletion_auc_predicted_confidence']:.4f} | {z['std_deletion_auc']:.4f} |")
parts += ['### 3.6 Explainability comparison for the selected model',
 'BioCLIP ViT-B/16 linear probe was selected for explanation analysis using source-validation macro-F1 alone (0.9723); target results did not determine this choice. Integrated Gradients, input saliency, and Occlusion were evaluated on the same stratified 12-image Europe subset. Lower deletion-AUC means predicted-class confidence decreased faster as highly attributed pixels were replaced by the normalized mean-color baseline. This perturbation-specific diagnostic is not a measure of agronomic value or causal fidelity.',
 figure(fig_no,'Mean deletion AUC for three explanation methods on the selected BioCLIP probe (n=12); error bars indicate ±1 SD. Lower is better.','XAI deletion-AUC comparison chart','xai_deletion_auc.png'),
 '**Table 9. Target-sample explanation faithfulness diagnostic.**', '\n'.join(xai_rows)]
fig_no+=1

parts += ['### 3.7 Comparative analysis',
 'Table 8 places all five evaluation conditions side by side after the model-specific analyses. BioCLIP linear probe is the strongest tested condition on this holdout; MobileNetV3-Large is the strongest CNN. Balanced accuracy and macro-F1 are included so performance is not judged by overall accuracy alone.',
 '**Table 8. Comparative performance on the European holdout (n=234).**']
comparison_rows=['| Model / condition | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro-F1 | Source-validation macro-F1 |','|---|---:|---:|---:|---:|---:|---:|']
for key,label in [('ResNet18','ResNet18, supervised'),('EfficientNet-B0','EfficientNet-B0, supervised'),('MobileNetV3-Large','MobileNetV3-Large, supervised'),('BioCLIP-B/16 linear probe','BioCLIP ViT-B/16, linear probe'),('BioCLIP zero-shot','BioCLIP ViT-B/16, prompt-only')]:
 r=row(key); val=f'{100*r.source_validation_macro_f1:.2f}%' if pd.notna(r.source_validation_macro_f1) else 'N/A'
 comparison_rows.append(f"| {label} | {100*r.accuracy:.2f}% | {100*r.balanced_accuracy:.2f}% | {100*r.macro_precision:.2f}% | {100*r.macro_recall:.2f}% | {100*r.macro_f1:.2f}% | {val} |")
parts.append('\n'.join(comparison_rows))
parts += ['**Synthesis and link to the Introduction.** The Introduction identified a gap between strong within-dataset crop/weed scores and limited evidence for target-continent evaluation without target examples. This experiment addresses that gap narrowly: for four known taxa and one North America-to-Europe split, both source-trained BioCLIP and prompt-only BioCLIP transfer well on this sample, while the three ImageNet CNNs score lower and show more maize/ragweed confusions. Thus, the results support the study objectives of (i) constructing a geographically held-out, target-example-free evaluation, (ii) comparing CNNs with a biological vision-language foundation model, and (iii) reporting per-class errors and a model-specific XAI diagnostic. They do **not** establish universal cross-continental performance, identify unseen species, or prove the source of the observed differences. The small test set, possible BioCLIP pretraining overlap, source-validation/target domain mismatch, and unequal class support constrain interpretation.']
results_block='\n\n'.join(parts)
text=ARTICLE.read_text(encoding='utf-8')
start=text.index('## 3. Results and Analysis')
end=text.index('## 4. Discussion',start)
prefix=text[:start].replace('Table 6','Table 10')
suffix=text[end:].replace('Table 6','Table 10')
suffix=suffix.replace('Table 10 summarizes seven recent agricultural vision and adaptation studies. Their label spaces, geographic coverage, test protocols, metrics, and use of target-domain data differ, so the reported values are contextual evidence rather than a common ranking.\n\n','')
text=prefix+results_block+'\n\n'+suffix
# In the Discussion, restore references to the new overall model comparison where applicable.
text=text.replace('The comparisons in Table 10 are not a common leaderboard.', 'The model comparison in Table 8 is not a cross-study leaderboard. The seven-study literature comparison in Table 10 is also not a common leaderboard.')
ARTICLE.write_text(text,encoding='utf-8')
print('Wrote',ARTICLE,'with',fig_no-1,'figures and',table_no-1,'model-specific classification-report tables')
