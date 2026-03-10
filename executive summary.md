# BRAVE-Net: Burg Residual Augmented Vision Transformer for Interpretable Parkinson’s Disease Detection from Dysarthric Speech

| Field                  | Details                                                                              |
| :--------------------- | :----------------------------------------------------------------------------------- |
| **Supervisor**         | Dr. Md. Ohidujjaman                                                                  |
| **Student Researcher** | MD Arif Faysal Nayem                                                                 |
| **Target Journals**    | IEEE Access (Q1, IF ~3.9) \| Biomedical Signal Processing and Control (Elsevier, Q1) |
| **Submission Type**    | Fast-Track Original Research Article                                                 |

---

## 1. Executive Summary

This research proposes a novel **“Signal-First”** paradigm for diagnosing Parkinson’s Disease (PD) from voice data. Existing AI classifiers fail on dysarthric speech because they treat pathological vocal irregularities as noise to be removed—discarding the very biomarkers that carry diagnostic value. We instead apply the **Burg-Based Linear Prediction with Residual Estimation method (Ohidujjaman et al., 2026)** to mathematically characterise and preserve these irregularities prior to classification.

The restored signals are then converted to high-resolution Mel-Spectrograms and classified using a fine-tuned **Vision Transformer (ViT-B/16)** trained via transfer learning. Clinical interpretability is ensured through **Grad-CAM heatmaps** that localise which spectro-temporal regions drive each prediction, allowing clinicians to verify that the model attends to known PD markers such as glottal tremors and hypophonia rather than artefacts.

---

## 2. Problem Statement & Research Gap

### 2.1 Clinical Challenge
Parkinsonian dysarthria produces two acoustically distinctive features: **hypophonia** (reduced vocal intensity) and **glottal tremors** (4–8 Hz involuntary oscillations of the larynx). Conventional pre-processing pipelines apply standard noise reduction, which attenuates precisely these features. The resulting signals are cleaner by conventional signal metrics but informationally impoverished for PD diagnosis.

### 2.2 Technical Gap in Existing Literature
The dominant approach in the literature (e.g., *Hassan et al., 2019*) extracts hand-crafted features such as MFCCs from raw audio and feeds them into shallow classifiers including Decision Trees and Random Forests. These pipelines have two structural limitations:
1. They operate on the signal surface and cannot model the **physics of the human vocal tract**.
2. They provide no mechanism for **clinical interpretability**, which is an increasingly hard requirement for regulatory acceptance of medical AI.

### 2.3 Research Hypothesis
We hypothesise that the **residual error term** in Linear Predictive Coding—the component discarded by standard models as unpredictable noise—encodes the primary biomarkers of PD-related vocal dysfunction. Estimating and selectively amplifying this residue using the *2025 Ohidujjaman Method* before deep feature extraction should yield a measurable and consistent improvement in classification accuracy over pipelines that operate on unrestored audio.

---
## 3. Proposed Methodology

The framework is composed of four sequential stages, each designed to address a specific technical limitation of prior work.

### Stage 1 — Physics-Informed Signal Restoration
We apply the **Modified Burg Method with Residual Estimation** (Equations 5–7, *Ohidujjaman et al., Results in Engineering, 2025*). rather than performing conventional denoising, the algorithm reconstructs spectral information in frequency bands associated with glottal tremors. Conceptually, this treats dysarthric signal degradation as a form of packet loss and applies a physics-grounded restoration model derived from vocal tract acoustics. The output is a set of spectrally enhanced waveforms that retain—and in fact amplify—pathologically significant signal components.

### Stage 2 — Visual Feature Representation
Each restored waveform is converted to a high-resolution **Mel-Spectrogram** (128 mel bands, hop length 256, window size 1024). This visual representation encodes both temporal dynamics and frequency-domain structure simultaneously, enabling the use of state-of-the-art computer vision architectures that have been extensively validated on texture and pattern recognition tasks.

### Stage 3 — Transfer Learning via Vision Transformer
We use a **ViT-B/16 model** pre-trained on ImageNet-21k as the classification backbone.
* To prevent overfitting given the limited size of the TORGO dataset, all multi-head self-attention blocks are **frozen** during training.
* Only the **MLP classification head** is fine-tuned, along with the final two transformer encoder layers via gradual unfreezing.

This strategy preserves the model’s generalised texture-recognition capabilities while adapting the decision boundary to PD-specific spectrogram patterns.

### Stage 4 — Clinical Interpretability via Grad-CAM
**Gradient-weighted Class Activation Mapping (Grad-CAM)** is applied to the final transformer block to generate per-sample saliency heatmaps. These heatmaps are overlaid on the original Mel-Spectrograms and evaluated against known PD vocal biomarkers. A positive result requires that the model’s attention consistently localises to frequency bands and temporal windows consistent with glottal tremors (100–300 Hz, 4–8 Hz modulation). This validation step distinguishes genuine clinical signal from spurious correlations with recording artefacts or speaker identity.

---

## 4. Comparative Baselines & Ablation Study

To isolate the contribution of each component, we conduct a full ablation study across three configurations. Performance is reported as mean ± standard deviation across a **leave-one-speaker-out cross-validation** scheme, which is mandatory given the small number of dysarthric speakers in TORGO.

| Configuration | Signal Input | Features | Classifier | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 1** | Raw Audio | MFCCs (13 coefficients) | Random Forest | Replicates *Hassan et al., 2019* |
| **Baseline 2** | Raw Audio | Mel-Spectrogram | ResNet-18 | Standard deep learning baseline |
| **Baseline 3** | Raw Audio | Mel-Spectrogram | ViT-B/16 | Isolates transformer contribution |
| **Proposed** | **Burg-Restored Audio** | **Mel-Spectrogram** | **ViT-B/16 (fine-tuned)** | **Full framework** |

The critical comparison is between **Baseline 3** and the **Proposed** method. Any accuracy gain attributable to the Burg restoration step, holding the model architecture constant, directly supports the central hypothesis. A target improvement of **+5 to +8 percentage points in F1-score** is anticipated based on prior signal restoration work in related pathology detection tasks.

---

## 5. Datasets & Software Stack

### 5.1 Datasets
| Dataset | Description | Role |
| :--- | :--- | :--- |
| **TORGO Database** | Dysarthric speech corpus, University of Toronto. 7 dysarthric speakers + 7 healthy controls. | Primary training & evaluation |
| **MDVR-KCL** | Parkinson’s voice dataset, King’s College London. Sustained phonation and read speech tasks. | Cross-dataset generalisation validation |

### 5.2 Software Stack
| Component | Tools & Libraries |
| :--- | :--- |
| **Signal Processing** | Python: `librosa`, `scipy`, `numpy` \| MATLAB: Burg Method implementation (*Ohidujjaman et al., 2025 codebase*) |
| **Deep Learning** | PyTorch, `timm` (ViT-B/16), HuggingFace Transformers |
| **Interpretability** | `pytorch-grad-cam`, `matplotlib`, `seaborn` |
| **Experiment Tracking** | Weights & Biases (`wandb`) for hyperparameter logging and reproducibility |
| **Statistical Analysis** | `scipy.stats` (McNemar’s test for pairwise model comparison, 95% CI bootstrapping) |

---

## 6. Evaluation Protocol

Given the clinical context and class imbalance typical in pathology datasets, accuracy alone is insufficient. The following metrics will be reported for all configurations:

1. **Sensitivity (Recall) and Specificity** — primary clinical metrics
2. **F1-Score** — primary comparative metric for ablation study
3. **Area Under the ROC Curve (AUC-ROC)**
4. **Matthew’s Correlation Coefficient (MCC)** — robust to class imbalance
5. **McNemar’s Test** — for statistical significance of pairwise comparisons

All metrics are computed under **leave-one-speaker-out cross-validation**. Speaker identity is strictly held out to prevent data leakage and ensure that the model cannot exploit speaker-level acoustic characteristics unrelated to PD pathology.

---

## 7. Execution Roadmap

| Phase                                      | Activities                                                                                                                                                               | Deliverable                                                                   |
| :----------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| **Week 1: Data & Signal Processing**       | Download and preprocess TORGO and MDVR-KCL. Implement and validate Burg Residual Estimation pipeline. Generate matched pairs of raw vs. restored audio for each speaker. | Processed dataset; restoration quality metrics (SNR, spectral flatness delta) |
| **Week 2: Baseline Experiments**           | Implement and run Baseline 1 (MFCC + RF) and Baseline 2 (Raw + ResNet-18). Generate Mel-Spectrograms for all conditions. Establish performance floor.                    | Baseline results table; confirmed “win” from restoration step                 |
| **Week 3: Transformer & Interpretability** | Fine-tune ViT-B/16 on restored spectrograms. Run Baseline 3. Generate Grad-CAM heatmaps. Validate heatmap localisation against known PD biomarker frequencies.           | Final model; Grad-CAM visualisations; full ablation table                     |
| **Week 4: Manuscript**                     | Write manuscript to IEEE Access LaTeX template. Prepare figures. Internal review and revision. Submit.                                                                   | Camera-ready manuscript submission                                            |
|                                            |                                                                                                                                                                          |                                                                               |

---

## 8. Expected Scientific Contributions

This work makes three distinct contributions to the field:

1. **Cross-Domain Algorithm Repurposing.** This is the first study to apply a telecommunications-derived signal restoration algorithm (Burg Residual Estimation) to the specific problem of preserving pathological vocal biomarkers for PD diagnosis, bridging signal processing and clinical AI.
2. **Interpretable Clinical AI.** The Grad-CAM validation protocol provides a replicable methodology for confirming that deep learning models attend to medically meaningful signal features, not spurious correlations—a requirement for clinical deployment.
3. **Rigorous Small-Data Benchmarking.** The four-condition ablation study with leave-one-speaker-out cross-validation and statistical significance testing establishes a reproducible benchmark for future PD detection research on dysarthric corpora.

