# CLIPQ-Count

**Query-Guided Vision-Language Framework for Few-Shot Object Counting**
S. Kranthi Kumar Chowdary, S. Prabakeran, M. Saravanan · *IEEE INDICON 2025*
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FINDICON68490.2025.11392884-blue)](https://doi.org/10.1109/INDICON68490.2025.11392884)

Count objects of a **novel category** from a handful of exemplar boxes, steered by a **text query**.
Built on CLIP ViT-B/32 (Hugging Face `transformers`); only the last two vision blocks are fine-tuned.

## Method
```
image ──► CLIP vision ──► patch grid ──► Point-Guided Attention ──► pooled ──┐
                                             │                               ├─► SFE gate ─┐
text query ─► CLIP text ─────────────────────┼───────────────────────────────┘             ├─► fusion ─► count head  (log-count → count)
exemplar crops ─► CLIP vision ─► mean ───────┼─────────────────────────────────────────────┘
                                             └─► density head ─► 224×224 density map
```
* **Point-Guided Attention** – a 1×1 conv gate on CLIP patch features, switched on at annotated instance locations during training.
* **Self-Adaptive Feature Enhancement (SFE)** – projects image features into the text space and gates them with the query embedding.
* **Fusion + two heads** – a count head (trained in log space) and a density head; loss = Smooth-L1(count) + 0.5·Smooth-L1(log count) + 0.1·MSE(density).

## Repository layout
```
clipq_count/
  data.py       FSC-147 dataset, exemplar loading, text prompts, density maps, collate
  modules.py    PointGuidedAttention, SelfAdaptiveFeatureEnhancement
  model.py      CLIPCountingModel
  loss.py       CombinedCountingLoss
  engine.py     train / eval loops (AMP, grad clipping, cosine LR, early stopping)
  utils.py      seeding, device, batch transfer, logging
scripts/
  train.py      train on FSC-147, save best checkpoint, report test MAE / RMSE
  evaluate.py   evaluate a checkpoint on any split
experimental/
  hybrid_hct.py ViT (timm) hybrid used for the "ViT + CLIP" ablation
```

## Setup
```bash
pip install -r requirements.txt
```
Download [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) and arrange it as
```
FSC147/
  images_384_VarV2/
  annotation_FSC147_384.json
  Train_Test_Val_FSC_147.json
```

## Train and evaluate
```bash
python scripts/train.py --data-root FSC147 --shots 3 --epochs 10
python scripts/evaluate.py --data-root FSC147 --checkpoint checkpoints/clipq_count_best.pth --split test
```
Defaults: 3-shot, batch 8, AdamW lr 1e-4, cosine schedule, early stopping (patience 3). See the reproducibility note for the paper configuration.

## Results (FSC-147 test split)
| Configuration | MAE | MAPE (%) | Improvement |
|---|---|---|---|
| Baseline CLIP | 36.60 | 52.10 | – |
| ViT + CLIP | 30.50 | 47.50 | 16.7 % |
| ViT + CLIP + SAFE | 28.80 | 46.00 | 21.3 % |
| ViT + Point-Guided Attention | 25.90 | 43.50 | 29.3 % |
| **CLIPQ-Count (full)** | **22.70** | **41.20** | **38.0 %** |

Numbers are from the paper (Table I). CLIPQ-Count is a lightweight head on top of CLIP, so its absolute MAE is well above
detection-based open-world counters such as CountGD (MAE ≈ 4); the contribution is the query-guided design and its
low overhead, not state-of-the-art accuracy. Failure cases: very dense scenes and small objects.

## Reproducibility note
The paper's training setup was 5-shot, 50 epochs, batch 16, AdamW lr 1e-4, weight decay 1e-4, 10-epoch linear warm-up,
early-stopping patience 15, dropout 0.1, on an NVIDIA A100. The released scripts default to a lighter configuration
(3-shot, 10 epochs, batch 8) that fits a single consumer GPU. To approach the paper setting:
```bash
python scripts/train.py --data-root FSC147 --shots 5 --epochs 50 --batch-size 16 --weight-decay 1e-4 --patience 15
```
Linear warm-up is not implemented in this release. `experimental/hybrid_hct.py` is the ViT (timm) hybrid used for the
"ViT + CLIP" ablation rows.

## Citation
```bibtex
@inproceedings{chowdary2025clipqcount,
  title     = {CLIPQ-Count: A Query-Guided Vision-Language Framework for Few-Shot Object Counting},
  author    = {Chowdary, S. Kranthi Kumar and Prabakeran, S. and Saravanan, M.},
  booktitle = {IEEE INDICON},
  year      = {2025},
  doi       = {10.1109/INDICON68490.2025.11392884}
}
```

## License
MIT
