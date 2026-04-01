# Fruit Quality Classification

> Automated fruit quality grading using deep learning and computer vision — with Grad-CAM interpretability

A computer vision pipeline that classifies fruit images into three quality grades — **Good**, **Imperfect**, and **Bad (spoiled)** — across 16 fruit varieties. Built to handle the real-world messiness of agricultural image data: severe class imbalance, limited samples for rare classes, and the need for human-interpretable predictions.

---

## Dataset

1,484 images sourced from the **Harvard Dataverse**, covering 16 fruit categories including apples, strawberries, lemons, and pomegranates. Labels are extracted programmatically from image filenames.

| Quality Class | Share |
|---------------|-------|
| Good | 51.48% |
| Imperfect | 43.53% |
| Bad (spoiled) | 4.99% |

The spoiled class at under 5% is the central challenge — a model that ignores it entirely still achieves ~95% accuracy, which is useless in practice.

---

## Preprocessing Pipeline

All images resized to **224×224**, converted to RGB, and normalised with ImageNet statistics.

**Augmentation (training set only):**
- Random rotations and horizontal flips
- Colour jittering (brightness, contrast, saturation)
- Random resized crops

**Split:** 70% train / 15% validation / 15% test — stratified to preserve class proportions across all subsets.

---

## Models

Two architectures compared via transfer learning.

### Baseline — ResNet18 (feature extractor)
Convolutional layers frozen; only the classification head trained.

- Test accuracy: **90%**
- Weakness: poor recall on the *bad* class, which the frozen features largely ignore

### Final model — EfficientNet-B0 (two-stage fine-tuning)
Stage 1: train classification head with frozen backbone.
Stage 2: unfreeze and fine-tune the full network at a lower learning rate.

**Weighted cross-entropy loss** applied to penalise misclassification of the minority *bad* class — the key intervention that drove meaningful improvement.

---

## Results

| Metric | ResNet18 | EfficientNet-B0 |
|--------|----------|-----------------|
| Validation accuracy | — | 94% |
| Test accuracy | 90% | 91% |
| Bad class detection | Weak | Significantly improved |

The accuracy gap between models is modest; the real gain is in **recall on spoiled fruit**, where the baseline fails and the final model delivers actionable detections.

---

## Interpretability — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) applied to visualise where the network attends when making quality decisions.

The model consistently focuses on:
- Surface texture and discolouration patches
- Shape irregularities and edges
- Localised spoilage regions

Background elements are largely ignored, confirming the model has learned visually meaningful features rather than dataset artefacts — an important check before deploying in any real quality-control pipeline.

---

## Limitations & Next Steps

- **Dataset expansion** — under-represented classes (some fruit types have < 20 samples) limit generalisation
- **Multi-task learning** — jointly predict fruit type and quality in a single forward pass
- **Edge deployment** — optimise via quantisation or pruning for integration into agricultural sorting hardware

---

## Stack

`Python` `PyTorch` `EfficientNet-B0` `ResNet18` `Grad-CAM` `transfer learning` `computer vision` `image classification`
