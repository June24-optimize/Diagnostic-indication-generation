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
- **Primary Dataset**: Private Low Field MRI dataset (requires annotation)

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



# Function to create a flowchart without arrows
def draw_flowchart_no_arrows():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Define positions for each stage in the flowchart
    positions = {
        "Input": (5, 11),
        "Preprocessing": (5, 9.5),
        "Feature Extraction": (5, 8),
        "Text Generation": (5, 6.5),
        "NLP Refinement": (5, 5),
        "Output": (5, 3.5),
        "Deployment": (7.5, 2),
        "Evaluation": (2.5, 2)
    }

    # Define labels for each stage
    labels = {
        "Input": "MRI Scan Input\n(DICOM/NIfTI)",
        "Preprocessing": "Preprocessing & Segmentation\n(Normalization, Enhancement)",
        "Feature Extraction": "Feature Extraction\n(Vision Transformer - ViT)",
        "Text Generation": "Text Generation\n(Transformer Decoder)",
        "NLP Refinement": "Clinical NLP Refinement\n(BioBERT)",
        "Output": "Final Diagnostic Indication",
        "Deployment": "Deployment\n(FastAPI, AWS SageMaker)",
        "Evaluation": "Evaluation Metrics\n(BLEU, ROUGE, F1-Score)"
    }

    # Draw rectangles for each stage with rounded edges (without arrows)
    for key, pos in positions.items():
        ax.add_patch(FancyBboxPatch((pos[0] - 1.8, pos[1] - 0.4), 3.6, 0.8,
                                     boxstyle="round,pad=0.3", edgecolor="black",
                                     facecolor="lightblue", linewidth=2))
        ax.text(pos[0], pos[1], labels[key], fontsize=10, ha="center", va="center", fontweight="bold")

    plt.title("Flowchart: AI-Powered Brain MRI Diagnostic System (No Arrows)", fontsize=12, fontweight="bold")
    plt.show()

# Call function to draw the flowchart without arrows
draw_flowchart_no_arrows()

