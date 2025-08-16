# Brain MRI Diagnostic Indication Generation Model

## Overview
This system generates **diagnostic reports** and **VQA answers** from 3D medical scans (CT/MRI).  
It fuses lesion/tissue **segmentations** with a **Med3DVLM** backbone to preserve long-range 3D context while steering the language model toward clinically relevant regions.

**Core idea:**  
Efficient 3D encoder → dual-stream **MLP-Mixer projector** with segmentation-aware fusion → **LLM** (Qwen2.5-7B-Instruct via LoRA) → report/VQA.  
An optional **retrieval step** grounds text in similar prior cases to reduce hallucinations.

---
## Inputs & Outputs
**Inputs**
- 3D volume (NIfTI/DICOM)
- Segmentation masks:
  - **Anomaly masks** (tumor/stroke) — SynthSeg / U-Net
  - **Tissue masks** — SynthSeg
- Optional metadata (modality, phase, site)
**Outputs**
- Structured **diagnostic report** (Findings/Impression) with **uncertainty** and **evidence tags**
- **VQA** answers (open-ended and closed-ended)

---
## Pipeline
1. **Preprocess**
   - Resample to `128 × 256 × 256`
   - Intensity normalize (clip HU for CT)
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
- **Retrieval**: Recall@K (R@1, R@5, R@10)
- **Report/VQA (open-ended)**: BLEU, ROUGE, METEOR, BERTScore
- **VQA (closed-ended)**: Accuracy
- **Additional**: Uncertainty calibration (ECE), entity grounding coverage


## Model Architecture

![Flowchart](/framework.png)

## Dataset
- **Primary Dataset**: Private Low Field MRI dataset (requires annotation), xxxx images for xxxx unhealthy subjects.

## Sample Inference Result

Below is an example of the model's generated diagnostic indication based on a Brain MRI input.

### **Input MRI Scan**
![Brain MRI Example](/RS036.png)

### **Generated Indication**
"Potential ischemic lesion detected in the frontal lobe region. The lesion appears hyperintense compared to surrounding tissues, suggesting an acute or subacute infarct. Clinical correlation with patient history of transient ischemic attacks (TIAs) is recommended."


### **True Indication (For Comparison)**
"Ischemic stroke. Location of stroke not confirmed. May be frontal lobe. Patient also had subsequent TIAs."


The model-generated output closely aligns with the true indication, demonstrating its capability in detecting abnormalities.


### **Input MRI Scan**
![Brain MRI Example](/RS043.png)

### **Generated Indication**
"Suspicious vascular abnormality detected in the cerebellar region. The hyperintense area suggests a possible aneurysmal dilation of blood vessels. Urgent further imaging with MRA (Magnetic Resonance Angiography) is recommended to confirm diagnosis and assess risk of rupture."

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


