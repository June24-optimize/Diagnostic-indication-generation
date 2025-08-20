# Diagnostic Indication Generation Model

## Overview
This system generates **diagnostic reports** and **VQA answers** from 3D medical scans (brain MRI as an example).  
It fuses lesion/tissue **segmentations** with a **Med3DVLM** backbone to preserve long-range 3D context while steering the language model toward clinically relevant regions.

**Motivation:**  
Making diagnostic decisions is a time-consuming process, and it is prone to variability across different doctors and hospitals. Our project aims to automate part of this process: generate structured diagnostic indications directly from medical images and reports. Structured outputs can support radiologists, speed up clinical workflows.

**Core idea:**  
Efficient 3D encoder → dual-stream **MLP-Mixer projector** with segmentation-aware fusion → **LLM** (Qwen2.5-7B-Instruct via LoRA) → report/VQA.  
An optional **retrieval step** grounds text in similar prior cases to reduce hallucinations.

---
## Inputs & Outputs
**Inputs**
- 3D volume (NIfTI/DICOM)
- Segmentation masks:
  - **Anomaly masks** (tumor/stroke) — SynthSeg with a conventional anomaly detection method
  - **Tissue masks** — SynthSeg
- Optional metadata (modality, phase, site)
  
**Outputs**
- Structured **diagnostic report** (Findings/Impression) with **uncertainty** and **evidence tags**
- **VQA** answers (open-ended and closed-ended)

---
## Pipeline
1. **Preprocess**
   - Resample to `128 × 256 × 256`
   - Intensity normalize (rescaled for MRI)
   - Run segmentation model → anomaly & tissue probability maps
2. **ROI Token Derivation**
   - Mask-aware pooling from encoder feature maps
   - ROI descriptors: centroid, bbox, size, slice range
   - Produces **segmentation-aware ROI tokens**
3. **Image Encoder**
   - Efficient 3D feature extractor (decomposed depthwise convolutions)
   - Two token streams:
     - High-level tokens (`32 × 768`)
     - Low-level tokens (`256 × 384`)
4. **Multi-Modal Projector (Dual MLP-Mixer + Seg-Aware Fusion)**
   - Two parallel **MLP-Mixer** stacks (low/high hybrid)
   - Fusion steps:
     - Concatenate ROI tokens with low/high tokens
     - **Mask gating**: drop low-importance tokens
     - **Token reweight**: boost lesion-relevant tokens
5. **(Optional) Multimodal Retrieval**
   - Build SigLIP image/text index offline
   - Retrieve top-K similar cases/reports at inference
6. **LLM Decoder**
   - **Qwen2.5-7B-Instruct** + LoRA adapters (base LLM frozen)
   - Inputs: fused tokens (+ optional retrieval evidence)
   - Outputs: diagnostic report + VQA answers


## Model Architecture

![Flowchart](/framework.png)


---
## Training Stages
1. **Contrastive Pretraining**
   - Image Encoder + ClinicalBERT
   - SigLIP loss for 3D image ↔ report alignment
2. **Projector Pretraining**
   - Freeze encoders, train dual MLP-Mixer projector
   - Teacher-forced LM loss with LLM frozen
3. **VLM Fine-Tuning**
   - Train projector + LoRA only
   - Mix report generation and VQA tasks
   - Include retrieval grounding

---
## Inference Flow
1. Preprocess & segment input volume
2. Image Encoder → low/high tokens + seg-ROI tokens
3. Projector (dual Mixer + seg fusion) → fused tokens
4. (Optional) Retrieve evidence via SigLIP index
5. LLM (+LoRA) generates:
   - Diagnostic report (with uncertainty/evidence tags)
   - VQA answers

---
## Key Design Choices
- **Image Encoder** — scales to large tiles while preserving 3D context
- **SigLIP** — better alignment in small medical datasets
- **Dual MLP-Mixer** — preserves both low-level detail & high-level semantics
- **Segmentation-aware fusion** — focuses LLM on clinically relevant regions
- **Retrieval grounding** — reduces hallucinations, improves rare-case handling
- **Normalization** — BN in segmentation heads; LayerNorm in Transformer/LLM paths

---
## Evaluation Metrics
- **Datasets**
- Primary dataset: Low-field brain MRI (xxxx volumes; annotated with lesion/tissue labels + diagnostic reports).
- External validation: Public M3D dataset (subset of CT/MRI scans with paired reports and VQA)
- Splits: 70% training, 15% validation, 15% testing.
- Preprocessing: all volumes resampled to 128 × 256 × 256; segmentation masks from SynthSeg and anomaly detection pipeline.
- **Baselines**
- 2D CLIP-style VLM: Slice-level encoder + ClinicalBERT (contrastive only).
- M3D-LaMed: CLIP-pretrained 3D encoder with linear projector
- Med3DVLM: state-of-the-art 3D VLM baseline
- Ours: Segmentation-aware fusion + retrieval grounding.

- **Image–Text Retrieval**: 
- Baseline 2D VLM: R@1 = 12%; M3D-LaMed: R@1 = 19%; Med3DVLM: R@1 = 61%; Ours (segmentation-aware + retrieval): R@1 = 67%, R@5 = 90%, R@10 = 95%.
- **Diagnostic Report Generation**: BLEU, ROUGE, METEOR, BERTScore
- Baseline: BLEU = 12.8, METEOR = 13.5;
- M3D-LaMed: BLEU = 15.1, METEOR = 14.3;
- Med3DVLM: BLEU = 36.9, METEOR = 36.4;
- Ours: BLEU = 39.2, METEOR = 38.1, BERTScore = 89.2.
- **VQA**: Closed-ended (accuracy) and open-ended (METEOR, BLEU).
- M3D-LaMed: Closed-ended acc = 75.8%; Open-ended METEOR = 33.6;
- Med3DVLM: Closed-ended acc = 79.9%; Open-ended METEOR = 36.8;
- Ours: Closed-ended acc = 82.4%; Open-ended METEOR = 38.9%;
- **Additional**: Uncertainty calibration (ECE)
-  Baseline models: ECE ~ 0.22–0.25;
-  Ours: ECE = 0.14, showing better reliability in predictions.


## Sample Inference Result

Below is an example of the model's generated diagnostic indication based on a Brain MRI input.

### **Input MRI Scan**
![Brain MRI Example](/RS036.png)

### **Generated Indication**
Med3DVLM: “Stroke lesion” (location missing).
Ours: "Potential ischemic lesion detected in the frontal lobe region. The lesion appears hyperintense compared to surrounding tissues, suggesting an acute or subacute infarct. Clinical correlation with patient history of transient ischemic attacks (TIAs) is recommended."

### **True Indication (For Comparison)**
"Ischemic stroke. Location of stroke not confirmed. May be frontal lobe. Patient also had subsequent TIAs."


Our model produces both lesion type and likely location, closer to the true indication.


### **Input MRI Scan**
![Brain MRI Example](/RS043.png)

### **Generated Indication**
Med3DVLM: “Large mass lesion in the brain.”
Our: "Suspicious vascular abnormality detected in the cerebellar region. The hyperintense area suggests a possible aneurysmal dilation of blood vessels. Urgent further imaging with MRA (Magnetic Resonance Angiography) is recommended to confirm diagnosis and assess risk of rupture."

### **True Indication (For Comparison)**
"Cerebellar aneurysm"


The generated output correctly identifies a vascular abnormality in the cerebellum, aligning with the true diagnosis of cerebellar aneurysm.



### **Input MRI Scan**
![Brain MRI Example](/RS050.png)

### **Generated Indication**
"Hyperintense hemorrhagic lesion detected in the right parietal region, suggesting a ruptured arteriovenous malformation (AVM). Evidence of acute bleeding with possible prior intervention. Further angiographic evaluation recommended for vascular integrity assessment."

### **True Indication (For Comparison)**
"AVM rupture, fixes (Bleed was 'above right ear,' the fixed AVM was in parietal cortex)."


The generated output correctly identifies a hemorrhagic lesion in the parietal region with a likely AVM rupture, closely matching the true diagnosis. 


### **Input MRI Scan**
![Brain MRI Example](/RS085.png)

### **Generated Indication**
"Structural abnormality detected in the right temporal lobe, consistent with focal cortical dysplasia. Findings suggest a prior right temporal lobectomy, likely for epilepsy management. Further clinical correlation and seizure history review recommended."


### **True Indication (For Comparison)**
"Dysplasia & epilepsy, Right temporal lobectomy for epilepsy. Cortical dysplasia of unknown size & type in right temporal lobe."


The model correctly detects cortical dysplasia in the right temporal lobe and prior lobectomy, aligning with the true diagnosis of dysplasia and epilepsy.


