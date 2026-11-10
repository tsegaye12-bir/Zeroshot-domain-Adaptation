# Reproducibility supplement

## Study

**Cross-Continental Zero-Shot Domain Adaptation for Weed and Crop Species Identification Using Vision-Language Foundation Models**

This supplement accompanies the revised manuscript. Results are presented consistently by model: separate source-training/source-validation accuracy plot, separate weighted-loss plot, European confusion matrix, full per-class classification report (including macro and weighted averages), and interpretation. The final Results subsection compares all evaluation conditions. Prompt-only BioCLIP has no training curves because it is inference without a fitted model.

## Experiment summary

- Dataset: Project-AgML iNatAg-mini; four taxa shared across North America and Europe.
- Unique selected images: 577 total; 343 source-continent records; 234 Europe target records.
- Fit split: 274 North America training images and 69 North America validation images; uploader-grouped validation.
- Target policy: Europe is held out from model fitting, validation, checkpoint selection and prompts.
- Trained conditions: ResNet18, EfficientNet-B0, MobileNetV3-Large, and a frozen BioCLIP ViT-B/16 image encoder with a trained linear head. Each trained model ran 30 epochs. Prompt-only BioCLIP is a separate no-iNatAg-label baseline and was not trained for 30 epochs.
- Augmentation: online stochastic transforms. Stored unique-image counts remain unchanged after augmentation; each trained model sees 8,220 augmented training presentations over 30 epochs.
- Selected model: BioCLIP linear probe, based on source-validation macro-F1, not target performance.

## Reproduction notes

Scripts under `code/` cover candidate-data inspection, training, post-training XAI, individual accuracy/loss plot generation, manuscript restructuring and DOCX rendering. The training script expects iNatAg-mini source archives and a locally materialized four-class image subset; it writes to the project path used for the reported run. Obtain images and annotations from the public Project-AgML iNatAg-mini release and follow its current terms and any individual image-provider licenses. Source archives and observations are not bundled. The run used Python 3.12, PyTorch 2.7.1 CPU, four CPU threads, seed 42, and no CUDA. See `results/run_config.json` for transforms, optimizer settings, split policy and hardware notes.

To regenerate the revised article from a prepared experiment workspace, generate the per-model plots with `make_per_model_curves.py`, run `restructure_results_by_model.py` against the saved metrics/reports/histories, then render the Markdown with `render_article_docx.py`. The original experiment's source images and observation metadata are intentionally not distributed here.

This is a research pilot, not a common benchmark leaderboard or a deployment claim. The target has 234 images and four known species. Potential overlap between BioCLIP's large-scale biological pretraining images and the iNatAg-derived test observations was not excluded.

## Privacy and licensing

The supplement omits raw photographs, exact coordinates, uploader names, local file paths, the row-level split manifest, and image-level target prediction records. This avoids republishing third-party images or disclosing fine-grained observation metadata. Aggregate counts and model-level outputs are included. Model checkpoints are not redistributed; pretrained-model terms and dependencies should be checked directly with the original providers.

## Directory map

- `code/`: data-audit, training, XAI, plot-generation, Results-restructuring and document-rendering scripts.
- `results/`: aggregate counts, configuration and summary, metrics, per-class reports, separate accuracy/loss plots for every trained model, individual confusion matrices, learning curves and XAI aggregate measurements/chart.
