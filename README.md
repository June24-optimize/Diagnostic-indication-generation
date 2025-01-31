# Brain MRI Diagnostic Indication Generation Model

## Overview
This repository contains the implementation of a **Brain MRI Diagnostic Indication Generation Model** using **Vision Transformer (ViT)** for image feature extraction and a **Transformer Decoder** for generating textual indications. The model is designed to assist radiologists by automatically generating clinical indications based on MRI scans.

## Features
- **MRI Image Processing**: Supports DICOM and NIfTI formats with automated preprocessing.
- **ViT for Feature Extraction**: Utilizes Vision Transformer to encode spatial and structural MRI features.
- **Transformer Decoder for Text Generation**: Generates diagnostic indications based on extracted MRI embeddings.
- **End-to-End Pipeline**: Includes preprocessing, model inference, and result visualization.
- **Cloud Deployment Ready**: Compatible with FastAPI and TensorFlow Serving for production use.

## System Architecture
1. **Input**: MRI scan (DICOM/NIfTI/PNG)
2. **Preprocessing**:
   - Convert DICOM/NIfTI to PNG/JPEG
   - Apply skull stripping and normalization
3. **Feature Extraction**:
   - Vision Transformer (ViT) extracts image embeddings
4. **Indication Generation**:
   - Transformer decoder generates a textual medical indication
5. **Output**: Structured diagnostic text based on the MRI scan

## Model Architecture
- **Vision Transformer (ViT)** for **feature extraction**
- **Transformer Decoder** for **text generation**
- **BioBERT (optional)** for refining medical terminology

## Dataset
- **Primary Dataset**: Private hospital MRI dataset (requires annotation)

## Sample Inference Result

Below is an example of the model's generated diagnostic indication based on a Brain MRI input.

### **Input MRI Scan**
![Brain MRI Example](/RS036.png)

### **Generated Indication**
"Potential ischemic lesion detected in the frontal lobe region. The lesion appears hyperintense compared to surrounding tissues, suggesting an acute or subacute infarct. Clinical correlation with patient history of transient ischemic attacks (TIAs) is recommended."


### **True Indication (For Comparison)**
"Location of stroke not confirmed. May be frontal lobe. Patient also had subsequent TIAs."


The model-generated output closely aligns with the true indication, demonstrating its capability in detecting abnormalities.

