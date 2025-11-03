# 🌱 PlantDoctor 

**PlantDoctor ** is an innovative application that **diagnoses your plants** from a single photo, assigns a **health score**, detects anomalies, provides **interactive recommendations**, and explains predictions using **Explainable AI (XAI)**. This project combines **Computer Vision + NLP + LLM + MLOps** for a complete, interactive, and CV-impressive experience.

---

## ✨ Key Features

### Core Capabilities
- **🔍 Plant Identification**: Automatic recognition of plant species using CNN (ResNet18 / EfficientNet)
- **🐛 Anomaly Detection**: Detects diseases, yellowing leaves, pests, and water stress
- **💚 Visual Health Score**: 0–100 gauge with color gradient (green → yellow → red)
- **🖼️ XAI / Grad-CAM**: Heatmaps highlight regions of the plant influencing predictions

### Enhanced Experience
- **🎭 Plant Personality**: Fun descriptions based on appearance and health status
- **💡 Interactive Recommendations**: Quick tips with emojis (🌞💧🐛) for each plant
- **🤖 CREAI + LangChain RAG**: Personalized advice and retrieval-based reasoning from a mini plant knowledge base
- **🏆 Gamification & Leaderboard**: Track multiple plants, earn "Perfect Plant" badges, and see rankings
- **📊 Dashboard Visualizations**: Score gauges, heatmaps, leaderboards, and eco metrics charts
- **🔀 Multi-Modal Input**: Combines image analysis with text descriptions for more precise diagnostics

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Version Control** | Git + DVC |
| **Vision Model** | CNN (ResNet18 / EfficientNet) |
| **Explainable AI** | Grad-CAM / Score-CAM |
| **LLM + Recommendations** | CREAI + LangChain mini RAG |
| **Experiment Tracking** | MLflow |
| **API Backend** | FastAPI |
| **Frontend/Dashboard** | Streamlit |
| **Deployment** | Docker + Render / Railway |
| **Monitoring** | Evidently AI, logging, drift detection |
| **Testing** | PyTest (data, model, API) |

---

## 📂 Project Structure
```
plantdoctor/
│
├── data/                    # Raw & processed images + mini KB for RAG
├── notebooks/               # EDA, preprocessing, baseline models
├── src/
│   ├── data/
│   ├── features/
│   ├── models/              # CNN training, Grad-CAM, prediction
│   ├── llm/                 # CREAI recommendations + LangChain RAG
│   └── api/                 # FastAPI backend
├── dashboard/               # Streamlit app (score, heatmaps, leaderboard)
├── tests/                   # Unit tests
├── .github/
│   └── workflows/           # CI/CD GitHub Actions
├── Dockerfile
├── requirements.txt
├── mlflow.yaml
├── dvc.yaml
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/plantdoctor.git
cd plantdoctor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize DVC and Download Dataset
```bash
dvc pull
```

### 4. Launch the FastAPI Backend
```bash
uvicorn src.api.app:app --reload
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

---

## 🔄 MLOps Pipeline

1. **Preprocessing** → Image augmentation + text embeddings
2. **Model Training** → CNN for plant identification + anomaly detection
3. **XAI Integration** → Grad-CAM heatmaps for model interpretability
4. **Multi-Modal Fusion** → Vision + text analysis
5. **LLM Recommendations** → CREAI + LangChain mini RAG for advice & personality
6. **Experiment Tracking** → MLflow for metrics & versioning
7. **Deployment** → Dockerized with CI/CD via GitHub Actions
8. **Monitoring** → Drift detection, health scores, anomaly tracking

---

## 🎮 Gamification & Dashboard

- **Multi-Plant Tracking**: Monitor multiple plants per user
- **Achievement Badges**: Earn rewards for maintaining perfect plants
- **Leaderboards**: Rankings for healthiest and most vulnerable plants
- **Interactive Visualizations**: Health gauges, heatmaps, eco-metrics
- **Eco Metrics**: Environmental impact scores for each plant

---

## 📊 Datasets

The following datasets are used for training and validation:

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)

---

## 🎯 Project Goals

Deliver a **complete MLOps solution** for plant diagnosis and monitoring with:

- **Computer Vision** for image-based detection
- **Explainable AI** for model transparency
- **LLM + Retrieval** for intelligent recommendations
- **Gamification & Interactive Dashboard** for user engagement
- **Eco-aware metrics** for sustainability

---

## 👥 Project Maintainers

This project is maintained by:

- **Asma Daab** - [LinkedIn]([https://www.linkedin.com/in/asma-daab](https://www.linkedin.com/in/asma-daab-b449051b6/))
- **Tesnime Ellabou** - [LinkedIn]([https://www.linkedin.com/in/tesnime-ellabou](https://www.linkedin.com/in/tesnime-ellabou-3170981b8/))
For questions, suggestions, or permission requests, please contact the maintainers via LinkedIn or open an issue.

---

## 📝 License

**Proprietary License - Permission Required**

This project and its source code are the exclusive property of Asma Daab and Tesnime Ellabou.

**You may NOT:**
- Use this code in any project (personal or commercial)
- Copy, modify, or distribute this code
- Deploy or host this application
- Use any part of this code without explicit written permission

**To request permission:** Contact the maintainers via LinkedIn

Copyright © 2024 Asma Daab & Tesnime Ellabou. All rights reserved.

---

**Made with 💚 for plant lovers, ML enthusiasts, and CV-ready tech!**
