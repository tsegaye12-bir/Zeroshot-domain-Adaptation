#!/usr/bin/env python3
from pathlib import Path
import json, re
import pandas as pd

BASE=Path('/home/ubuntu/research'); OUT=BASE/'results'
DRAFT=BASE/'cross_continental_article_draft.md'; FINAL=BASE/'cross_continental_zero_shot_article.md'
metrics=pd.read_csv(OUT/'model_metrics.csv'); reports=json.loads((OUT/'classification_reports.json').read_text())
xai_path=OUT/'xai_deletion_summary.csv'
if not xai_path.exists() or not (OUT/'xai_comparison.png').exists():
    raise SystemExit('XAI outputs are missing; refusing to finalize with placeholders.')
xai=pd.read_csv(xai_path)
histories={p.stem.replace('_history',''):pd.read_csv(p) for p in OUT.glob('*_history.csv')}
NAME_MAP={
 'ResNet18':'ResNet18, supervised',
 'EfficientNet-B0':'EfficientNet-B0, supervised',
 'MobileNetV3-Large':'MobileNetV3-Large, supervised',
 'BioCLIP-B/16 linear probe':'BioCLIP ViT-B/16, linear probe',
 'BioCLIP zero-shot':'BioCLIP ViT-B/16, prompt-only'}

def metric(name): return metrics.loc[metrics.model==name].iloc[0]
def pct(v): return f'{100*float(v):.2f}'

# Model-level metrics.
metric_rows=[]
for key,label in NAME_MAP.items():
 r=metric(key); v=pct(r.source_validation_macro_f1) if pd.notna(r.source_validation_macro_f1) else 'N/A'
 metric_rows.append(f'| {label} | {pct(r.accuracy)}% | {pct(r.balanced_accuracy)}% | {pct(r.macro_precision)}% | {pct(r.macro_recall)}% | {pct(r.macro_f1)}% | {v}{"%" if v!="N/A" else ""} |')

# Full per-class reports from the actual completed classification reports.
class_names=['Zea mays (maize; crop)','Glycine max (soybean; crop)',
 'Ambrosia artemisiifolia (common ragweed; weed)',
 'Raphanus raphanistrum (wild radish; weed)']
class_rows=[]
for key,label in NAME_MAP.items():
 found={}
 for line in reports[key].splitlines():
  s=line.strip()
  for cname in class_names:
   if s.startswith(cname):
    vals=s[len(cname):].split()
    if len(vals)>=4: found[cname]=vals[-4:]
 for cname in class_names:
  if cname not in found: raise RuntimeError(f'Missing class report {key}/{cname}')
  p,r,f,support=found[cname]
  class_rows.append(f'| {label} | {cname} | {p} | {r} | {f} | {support} |')

best_epochs=[]
for k in ['ResNet18','EfficientNet_B0','MobileNetV3_Large','BioCLIP_B16_linear']:
 h=histories[k]; row=h.loc[h.val_macro_f1.idxmax()]
 best_epochs.append(f"{k.replace('_','-')} epoch {int(row.epoch)} (validation accuracy {row.val_accuracy:.4f}, macro-F1 {row.val_macro_f1:.4f})")

xai_rows=[f"| {r['method']} | {int(r['n_explanations'])} | {r['mean_deletion_auc_predicted_confidence']:.4f} | {r['std_deletion_auc']:.4f} |" for _,r in xai.iterrows()]
xai_sort=xai.sort_values('mean_deletion_auc_predicted_confidence')
xai_best=xai_sort.iloc[0]; xai_worst=xai_sort.iloc[-1]

text=DRAFT.read_text()
# Abstract results.
text=text.replace('[RESULTS: INSERT AFTER TRAINING COMPLETES.]',
 f"The BioCLIP ViT-B/16 linear probe achieved {pct(metric('BioCLIP-B/16 linear probe').accuracy)}% accuracy and {pct(metric('BioCLIP-B/16 linear probe').macro_f1)}% macro-F1 on Europe; prompt-only BioCLIP reached {pct(metric('BioCLIP zero-shot').accuracy)}% and {pct(metric('BioCLIP zero-shot').macro_f1)}%, while the best CNN, MobileNetV3-Large, reached {pct(metric('MobileNetV3-Large').accuracy)}% accuracy and {pct(metric('MobileNetV3-Large').macro_f1)}% macro-F1. These are pilot results on 234 held-out images. Possible overlap of BioCLIP's pretraining images with iNaturalist-derived test data was not excluded.")

# Results and figure paths.
text=text.replace('[RESULTS: INSERT 30-epoch metrics, selected validation checkpoints, and relative performance after training completes.]',
 f"All four trained configurations completed 30 epochs; best source-validation checkpoints are {', '.join(best_epochs)}. On the untouched Europe test set, BioCLIP's source-trained linear probe achieved {pct(metric('BioCLIP-B/16 linear probe').accuracy)}% accuracy, {pct(metric('BioCLIP-B/16 linear probe').balanced_accuracy)}% balanced accuracy, and {pct(metric('BioCLIP-B/16 linear probe').macro_f1)}% macro-F1. Prompt-only BioCLIP, which used no iNatAg labels for fitting, obtained {pct(metric('BioCLIP zero-shot').accuracy)}% accuracy and {pct(metric('BioCLIP zero-shot').macro_f1)}% macro-F1. The strongest CNN was MobileNetV3-Large ({pct(metric('MobileNetV3-Large').accuracy)}% accuracy; {pct(metric('MobileNetV3-Large').macro_f1)}% macro-F1); EfficientNet-B0 and ResNet18 scored {pct(metric('EfficientNet-B0').accuracy)}% and {pct(metric('ResNet18').accuracy)}% accuracy. Linear probing improved on prompt-only BioCLIP by {100*(metric('BioCLIP-B/16 linear probe').accuracy-metric('BioCLIP zero-shot').accuracy):.2f} percentage points in accuracy. Only the 2,052-parameter classification head was trained in the BioCLIP probe; the full encoder-plus-head has 149.6 million parameters. The CNN parameter counts are 11.18M (ResNet18), 4.01M (EfficientNet-B0), and 4.21M (MobileNetV3-Large).")
text=text.replace('[INSERT LINK TO accuracy_curves.png]', '![Source-train and source-validation accuracy over 30 epochs](/home/ubuntu/research/results/accuracy_curves.png)')
text=text.replace('[INSERT LINK TO loss_curves.png]', '![Source-train and source-validation loss over 30 epochs](/home/ubuntu/research/results/loss_curves.png)')
text=text.replace('| ResNet18, supervised | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |\n| EfficientNet-B0, supervised | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |\n| MobileNetV3-Large, supervised | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |\n| BioCLIP ViT-B/16, linear probe | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |\n| BioCLIP ViT-B/16, prompt-only | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | N/A |', '\n'.join(metric_rows))
text=text.replace('[RESULTS: INSERT PER-CLASS PRECISION, RECALL, F1 AND SUPPORT FOR EACH OF THE FOUR TRAINED MODELS AND PROMPT-ONLY BIOCLIP, FROM GENERATED CLASSIFICATION REPORTS.]',
 'The table reproduces precision, recall, F1, and support from every generated classification report (support totals 234 per model).')
text=text.replace('### 3.2 Per-class classification reports\n\nThe table reproduces precision, recall, F1, and support from every generated classification report (support totals 234 per model).',
 '### 3.2 Per-class classification reports\n\nThe table reproduces precision, recall, F1, and support from every generated classification report (support totals 234 per model).\n\n| Model / mode | Taxon | Precision | Recall | F1 | Support |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(class_rows))
text=text.replace('[RESULTS: INSERT OBSERVED EUROPE HOLDOUT CONFUSION MATRIX INTERPRETATION AND LINKS TO ALL GENERATED MODEL MATRICES.]',
 'The five matrices below are presented separately as requested. In the BioCLIP linear-probe matrix, 220/234 images lie on the diagonal: all 51 maize images are correct; the largest errors are five common-ragweed images predicted as maize and four soybean images predicted as maize. Prompt-only BioCLIP correctly classifies 215/234; its largest error is six common-ragweed images assigned to soybean. Among CNNs, common ragweed is frequently confused with maize (11 ResNet18, 19 EfficientNet-B0, and 13 MobileNetV3-Large images). These repeated confusions may reflect visual or context similarity, but this experiment cannot establish their cause. All matrices show raw image counts, with rows as true classes and columns as predictions.\n\n**Figure 3. ResNet18 European holdout confusion matrix (n=234).**\n\n![ResNet18 confusion matrix](/home/ubuntu/research/results/ResNet18_confusion_matrix.png)\n\n**Figure 4. EfficientNet-B0 European holdout confusion matrix (n=234).**\n\n![EfficientNet-B0 confusion matrix](/home/ubuntu/research/results/EfficientNet_B0_confusion_matrix.png)\n\n**Figure 5. MobileNetV3-Large European holdout confusion matrix (n=234).**\n\n![MobileNetV3-Large confusion matrix](/home/ubuntu/research/results/MobileNetV3_Large_confusion_matrix.png)\n\n**Figure 6. BioCLIP ViT-B/16 linear-probe European holdout confusion matrix (n=234).**\n\n![BioCLIP linear-probe confusion matrix](/home/ubuntu/research/results/BioCLIP_B_16_linear_probe_confusion_matrix.png)\n\n**Figure 7. BioCLIP ViT-B/16 prompt-only European holdout confusion matrix (n=234).**\n\n![BioCLIP prompt-only confusion matrix](/home/ubuntu/research/results/BioCLIP_zero_shot_confusion_matrix.png)')
text=text.replace('[RESULTS: INSERT SELECTED MODEL, SOURCE-VALIDATION SELECTION BASIS, XAI DELETION-AUC VALUES, VISUAL OBSERVATIONS AND IMAGE LINK.]',
 f"BioCLIP ViT-B/16 linear probe was selected because it achieved the highest source-validation macro-F1 ({metric('BioCLIP-B/16 linear probe').source_validation_macro_f1:.4f}); selection did not use target performance. On the stratified 12-image Europe explanation sample, {xai_best['method']} had the lowest mean deletion-AUC ({xai_best['mean_deletion_auc_predicted_confidence']:.4f}±{xai_best['std_deletion_auc']:.4f}), while {xai_worst['method']} had the highest ({xai_worst['mean_deletion_auc_predicted_confidence']:.4f}±{xai_worst['std_deletion_auc']:.4f}). Lower AUC indicates faster predicted-class confidence reduction as the method's most salient pixels are zeroed. This is a small, model- and perturbation-specific faithfulness diagnostic, not expert-validated interpretability.")
text=text.replace('[INSERT LINK TO xai_comparison.png]', '![Mean deletion AUC by explanation method; lower is better, error bars show plus/minus one standard deviation](/home/ubuntu/research/results/xai_deletion_auc.png)')
text=text.replace('| Integrated Gradients | [TBD] | [TBD] | [TBD] |\n| Input Saliency | [TBD] | [TBD] | [TBD] |\n| Occlusion | [TBD] | [TBD] | [TBD] |', '\n'.join(xai_rows))
text=text.replace('[RESULTS-DEPENDENT DISCUSSION: INSERT AFTER THE EXPERIMENT. INTERPRET THE ACTUAL MODEL RANKING, CLASS ERRORS, XAI DIAGNOSTICS, AND SAMPLE UNCERTAINTY. DO NOT GENERALIZE BEYOND THE FOUR-TAXON NORTH-AMERICA-TO-EUROPE PILOT.]',
 f"The experiment yields three bounded findings. First, BioCLIP's source-trained linear probe exceeded the best tested ImageNet CNN on this four-class Europe holdout by {100*(metric('BioCLIP-B/16 linear probe').accuracy-metric('MobileNetV3-Large').accuracy):.2f} percentage points in accuracy and {100*(metric('BioCLIP-B/16 linear probe').macro_f1-metric('MobileNetV3-Large').macro_f1):.2f} points in macro-F1. This descriptive comparison does not isolate the effect of pretraining: model capacity, optimization, label use and potential BioCLIP pretraining overlap differ. Second, prompt-only BioCLIP achieved {pct(metric('BioCLIP zero-shot').accuracy)}% accuracy without iNatAg supervised labels; source-label linear probing raised it to {pct(metric('BioCLIP-B/16 linear probe').accuracy)}%. Neither condition used European images for fitting, but the probe used 274 North American labels for the known classes. Third, macro-F1 (94.03% probe; 91.91% prompt-only) and balanced accuracy (94.30%; 92.31%) track overall accuracy, so the strong performance is not explained only by class counts. Error inspection shows that errors were not uniform: the CNNs often predicted maize for common ragweed, whereas the BioCLIP probe made only five such errors and correctly recognized all European maize images in this sample. Prompt-only BioCLIP most often confused ragweed with soybean. Such patterns can arise from appearance, background, observation conditions, or label noise; without controlled image analysis they remain hypotheses. XAI deletion AUC modestly favored {xai_best['method']} over {xai_worst['method']} on 12 images, but this small difference does not demonstrate that either method provides biologically faithful explanations.")
text=text.replace('[RESULTS-DEPENDENT INTERPRETATION: Compare this study\'s actual measurements to the seven rows without asserting numerical superiority unless task, label space, and protocol are genuinely matched.]',
 'The comparisons in Table 6 are not a common leaderboard. Local greenhouse and private maize-field classifiers [11], [14] report high scores under familiar collection conditions. The Korean-field UDA study [12] and ground-to-UAV transfer paper [17] target segmentation and use target imagery or related target data. Deng et al. [8] document cross-season performance drops and modest UDA recovery, highlighting the importance of test protocol. Singh et al. [18] show the value of target labels for detection, but report mAP rather than classification accuracy. The single-crop UAV VLM study [19] and AgroBench [6] show agricultural VLM performance depends on task and evaluation format; AgroBench specifically finds weed identification difficult for many open VLMs. CLIP cross-dataset classification [20] is closer in model family but uses dataset-defined splits rather than the coordinate-defined continental split here. Prompt-only and linear-probe BioCLIP results are encouraging; nevertheless, pretraining overlap between BioCLIP and iNaturalist-derived test observations could inflate the score. The experiment therefore tests a documented geographic holdout but cannot conclusively measure pretraining-independent continent generalization.')
# Insert measured current-study row into contextual table.
text=text.replace('| **This study** | 577 iNatAg-mini images; 4 taxa; 274 source-train, 69 source-validation, 234 Europe-only target-test; no target examples in fitting | [INSERT AFTER RUN] | Four species; small pilot; geographic holdout with known classes |',
 f"| **This study** | 577 iNatAg-mini images; 4 taxa; 274 source-train, 69 source-validation, 234 Europe-only target-test; no target examples in fitting | BioCLIP probe {pct(metric('BioCLIP-B/16 linear probe').accuracy)}% / {pct(metric('BioCLIP-B/16 linear probe').macro_f1)}% macro-F1; prompt-only {pct(metric('BioCLIP zero-shot').accuracy)}% / {pct(metric('BioCLIP zero-shot').macro_f1)}%; best CNN MobileNetV3 {pct(metric('MobileNetV3-Large').accuracy)}% / {pct(metric('MobileNetV3-Large').macro_f1)}% | Four species; small pilot; geographic holdout with known classes |")
text=text.replace('[RESULTS-DEPENDENT CONCLUSION: INSERT AFTER TRAINING. SUMMARIZE ONLY THE MEASURED SOURCE-VALIDATION AND EUROPE-TARGET RESULTS, AND NAME THE WINNER BY SOURCE-VALIDATION SELECTION.]',
 f"All model fits completed 30 epochs, and the BioCLIP ViT-B/16 linear probe was selected using source-only validation macro-F1 ({metric('BioCLIP-B/16 linear probe').source_validation_macro_f1:.4f}). On the 234-image European holdout, it achieved {pct(metric('BioCLIP-B/16 linear probe').accuracy)}% accuracy and {pct(metric('BioCLIP-B/16 linear probe').macro_f1)}% macro-F1; prompt-only BioCLIP achieved {pct(metric('BioCLIP zero-shot').accuracy)}% and {pct(metric('BioCLIP zero-shot').macro_f1)}%; and the best CNN, MobileNetV3-Large, achieved {pct(metric('MobileNetV3-Large').accuracy)}% and {pct(metric('MobileNetV3-Large').macro_f1)}%. The apparent advantage is promising but uncertain because the pilot is small and pretraining-image overlap was not tested. The principal contribution is a reproducible target-example-free geographic evaluation protocol, not a deployment-ready score or a claim about global accuracy.")
# Correct sequential figure labels and keep one caption per image.
text=text.replace('**Figure 4.** Input, Integrated Gradients, input-saliency, and occlusion maps', '**Figure 8.** Input, Integrated Gradients, input-saliency, and occlusion maps')
text=text.replace('**Figure 3.** Input, Integrated Gradients, input-saliency, and occlusion maps', '**Figure 8.** Input, Integrated Gradients, input-saliency, and occlusion maps')
text=text.replace("**Figure 8.** Input, Integrated Gradients, input-saliency, and occlusion maps for one held-out European image per taxon. Maps explain the model's predicted class; panel labels indicate whether the prediction is correct.", '**Figure 8.** Mean deletion AUC by explanation method for the selected BioCLIP probe (n=12; error bars show ±1 SD). Lower is better.')
text=text.replace(' ![', '\n\n![')
text=text.replace('These are pilot results on 234 held-out images. Possible overlap of BioCLIP\'s pretraining images with iNaturalist-derived test data was not excluded. This is a pilot test of zero-target-example geographic transfer for known species, not novel-species recognition; the image sample is small and the BioCLIP pretraining corpus may overlap iNaturalist-derived data. The results should therefore be interpreted as a leakage-aware benchmark probe rather than evidence of continent-wide generalization.',
 'These pilot findings represent zero-target-example geographic transfer for known species, not novel-species recognition. The sample is small, and possible overlap between BioCLIP pretraining data and iNaturalist-derived test records was not ruled out; the results therefore do not establish continent-wide generalization.')
text=text.replace('The supervised architecture comparison comprises four distinct model families. ResNet18, EfficientNet-B0, and MobileNetV3-Large use standard ImageNet-pretrained weights and replace the final classification layer with a four-class output.',
 'The supervised architecture comparison comprises four distinct model families. ResNet18 uses residual basic blocks and a four-way fully connected output; EfficientNet-B0 uses mobile inverted bottleneck blocks with squeeze-and-excitation; MobileNetV3-Large uses inverted residual blocks with squeeze-and-excitation and mobile-oriented nonlinearities. All three use standard ImageNet-pretrained weights and replace the final classification layer with a four-class output. BioCLIP ViT-B/16 is a 224-pixel vision transformer with 16×16 patch tokens and a contrastively aligned text encoder; its visual tower is held fixed in the linear-probe condition.')
text=text.replace('The table reproduces precision, recall, F1, and support from every generated classification report (support totals 234 per model).\n\n| Model / mode | Taxon | Precision | Recall | F1 | Support |',
 'The table reproduces precision, recall, F1, and support from every generated classification report (support totals 234 per model).\n\n**Table 4. Per-class European holdout classification report.**\n\n| Model / mode | Taxon | Precision | Recall | F1 | Support |')
text=text.replace('**Table 4. Target-sample explanation faithfulness diagnostic.**', '**Table 5. Target-sample explanation faithfulness diagnostic.**')
text=text.replace('**Table 6. Contextual comparison; metrics are not directly comparable across tasks.**', '**Table 6. Contextual comparison with seven recent studies; metrics are not directly comparable across tasks.**')
text=text.replace('and lists iNatAg as a multi-class crop/weed resource [1].', 'and lists iNatAg as a multi-class crop/weed resource [1]. Other widely used agricultural image resources include CropAndWeed, Weed25, and DeepWeeds; PlantDoc illustrates field-style imagery, although its task is plant-disease detection rather than weed taxonomy [9], [10], [23], [24].')
text=text.replace('The image count does not increase as a stored dataset: the unique-image counts before and after augmentation remain 577 overall and 274 in the train fold.', 'Before augmentation, the selected data contain 577 unique source/target images, including 274 training images; after augmentation, the stored unique-image counts remain 577 overall and 274 in training because transforms are generated online, not saved as duplicates.')
text=text.replace('### 4.2 Limitations\n\nThis is a small pilot', '### 4.2 Ethical considerations\n\nThis study reuses the public iNatAg-mini dataset and does not involve new human-subject recruitment, intervention, or collection of personally identifying information. Images and annotations remain attributable to their original dataset providers; the manuscript reports aggregated taxon, continent, and performance results and does not republish source photographs or precise observation coordinates. Users reproducing this work should follow dataset and image-provider terms, retain attribution, and avoid exposing sensitive location data. Weed and crop classifications are observational labels, not autonomous treatment advice; any field deployment should include agronomic oversight and consider non-target species and local ecological consequences.\n\n### 4.3 Limitations\n\nThis is a small pilot')
text=text.replace('BioCLIP ViT-B/16 is a 224-pixel vision transformer with 16×16 patch tokens and a contrastively aligned text encoder; its visual tower is held fixed in the linear-probe condition. BioCLIP ViT-B/16 is initialized from the public `imageomics/bioclip` checkpoint.', 'BioCLIP ViT-B/16 is a vision transformer with 16×16 patch tokens and a contrastively aligned text encoder, initialized from the public `imageomics/bioclip` checkpoint; its visual encoder is held fixed during the linear probe.')
text=re.sub(r'Table 6 contextualizes the experiment against seven recent works\..*?beside every score\.', 'Table 6 summarizes seven recent agricultural vision and adaptation studies. Their label spaces, geographic coverage, test protocols, metrics, and use of target-domain data differ, so the reported values are contextual evidence rather than a common ranking.', text, flags=re.S)
# Replace non-IEEE repository attribution.
text=text.replace('[1] R. B. and contributors, “Digital Agriculture Datasets,” GitHub repository, accessed Nov. 9, 2026.', '[1] ricber and contributors, “Digital Agriculture Datasets,” GitHub repository, accessed Nov. 9, 2026.')

unresolved=[line for line in text.splitlines() if '[TBD]' in line or '[RESULTS:' in line or '[RESULTS-' in line or 'INSERT AFTER' in line or 'INSERT LINK' in line]
if unresolved: raise SystemExit('Unresolved placeholders:\n'+'\n'.join(unresolved))
FINAL.write_text(text)
print(FINAL)
print('words',len(text.split()),'numbered references',len(re.findall(r'^\[\d+\]',text,re.M)))
