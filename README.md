# PlantDoctor

A comprehensive plant disease diagnosis system using MobileNetV2 and RAG (Retrieval-Augmented Generation) for contextual plant disease explanations.

## 🌟 Features

- **Vision-based Plant Disease Classification**: Uses MobileNetV2 for accurate identification of plant diseases from leaf images.  
- **End-to-End System**: React frontend, FastAPI backend, and Azure cloud hosting for scalable deployment.  
- **RAG System**: Retrieval-Augmented Generation provides human-readable explanations and treatment suggestions.  
- **Dataset Augmentation & Preprocessing**: Rotation, flipping, scaling, and normalization to enhance model performance.  
- **Experiment Tracking**: W&B (Weights & Biases) integration for monitoring training and performance metrics.  
- **Data Versioning**: DVC on Azure for dataset and model version control.  
- **Containerization**: Docker used for reproducible deployments.  
- **Model Benchmarking**: Compared Custom CNN, ResNet18, and MobileNetV2; MobileNetV2 selected for best performance.  
- **Interactive UI**: Upload images and receive disease predictions with contextual insights.  
- **Explainable AI**: Combines MobileNetV2 predictions with expert knowledge for decision support.  

## 🏗️ Architecture

### Vision Model
- **MobileNetV2**: Lightweight CNN optimized for plant disease classification.
  - Input: 224×224×3 images  
  - Output: Multiple plant disease classes  
  - Fine-tuning applied only on classification head for efficiency  

### RAG Pipeline
1. **Image Analysis**: MobileNetV2 predicts top disease classes with confidence scores.  
2. **Query Generation**: Structured search queries created from user input and predictions.  
3. **Knowledge Retrieval**: FAISS-based semantic search through plant disease database.  
4. **Response Generation**: Comprehensive outputs combining model predictions and retrieved knowledge.  

### System Components
- **Frontend**: React.js interface for image upload and results display.  
- **Backend**: FastAPI handles image requests, prediction, and RAG queries.  
- **Cloud Deployment**: Azure for scalable hosting of API and frontend services.  
- **Experiment Management**: W&B dashboards for tracking training runs.  
- **Data & Model Versioning**: DVC on Azure ensures reproducibility.  
- **Containerization**: Docker images for consistent deployment across environments.

## 📋 Technologies Used

| Category            | Technologies                                     |
|--------------------|-------------------------------------------------|
| Languages          | Python, JavaScript (React)                      |
| Deep Learning      | PyTorch, TensorFlow                             |
| Backend            | FastAPI                                         |
| Frontend           | React.js                                        |
| Cloud Hosting      | Microsoft Azure                                 |
| Data Visualization | Matplotlib, Seaborn                             |
| Experiment Tracking| Weights & Biases (W&B)                          |
| Data Versioning    | DVC on Azure                                   |
| Containerization   | Docker                                          |
| Explainability     | RAG-based contextual information retrieval      |

## Installation

1. Clone the repository:

```bash
[git clone https://github.com/yourusername/PlantDoctor.git
](https://github.com/tass25/PlantDoctor)cd PlantDoctor
```
2.Create a Python environment and install dependencies:
```python
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```
3.Use DVC to manage datasets and model checkpoints:
```bash
dvc pull
```
4.Start the FastAPI backend:
```bash
uvicorn app.main:app --reload
```
5.Start the React frontend:
```bash
cd frontend
npm install
npm start
```

## Dataset

- **Training images**: 70,295  
- **Validation images**: 17,572  
- **Total images**: 87,867  
- **Top classes**: Soybean_healthy, Apple_Apple_scab, Orange_Citrus_canker, etc.  

Dataset preprocessing includes normalization, augmentation (rotation, flipping, scaling), and resizing.

---

## Model Details

- **Architecture**: MobileNetV2  
- **Input Size**: 224×224×3  
- **Output Classes**: Multiple plant disease categories  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Cross-Entropy  
- **Training Metrics**: Accuracy, Precision, Recall, F1-Score

---

## Performance & Evaluation

- **Validation Accuracy**: 95%+ on top classes  
- **Confusion Matrix**: Ensures correct identification of similar diseases  
- **Explainability**: RAG provides human-readable context for model outputs

---

## Future Work

- Expand dataset to cover more plant species and rare diseases.  
- Implement **real-time disease detection** using mobile devices.  
- Integrate IoT sensors for environmental monitoring.  
- Add **multi-modal inputs** (images + soil & climate data) for improved predictions.  
- Deploy a **progressive web app (PWA)** for offline use in remote farms.

- [PlantVillage](https://plantvillage.psu.edu/) for dataset resources  
- [FAISS](https://github.com/facebookresearch/faiss) for semantic search  
