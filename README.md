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

## Dataset
- **Primary Dataset**: Private hospital MRI dataset (requires annotation)

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
