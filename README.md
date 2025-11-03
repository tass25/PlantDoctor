# 🌱 PlantDoctor 

**PlantDoctor ** is an innovative application that **diagnoses your plants** from a single photo, assigns them a **health score**, detects anomalies, and provides interactive recommendations. This project combines **Computer Vision + NLP + MLOps** for a complete and engaging experience.

---

## ✨ Key Features

### Core Capabilities
- **🔍 Plant Identification**: Automatic recognition of plant species
- **📖 Care Information**: Maintenance tips, watering schedules, and light exposure requirements
- **🐛 Anomaly Detection**: Identifies diseases, yellowing leaves, pests, and water stress
- **💚 Visual Health Score**: 0-100 gauge with color gradient (green → yellow → red)

### Enhanced Experience
- **🎭 Plant Personality**: Fun descriptions based on appearance and health status
- **💡 Interactive Recommendations**: Quick tips with emojis (🌞💧🐛) for each plant
- **🏆 Gamification & Leaderboard**: Multi-plant tracking, "Perfect Plant" badges, average scores, and rankings
- **📸 Time-Lapse AI**: Monitor plant health evolution through multiple photos over time
- **🔀 Multi-Modal Input**: Combines image analysis with text descriptions for precise diagnostics
- **🌍 Eco Mode**: Environmental score based on water and light requirements
- **🤖 Integrated LLM**: Free generation of personalized recommendations and advice

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Version Control** | Git + DVC |
| **Vision Model** | CNN (ResNet18 / EfficientNet) |
| **NLP Model** | Free LLM (GPT4All, MPT-7B-Instruct) |
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
├── data/                    # Datasets (raw, processed, external)
├── notebooks/               # EDA, preprocessing, baseline models
├── src/                     # Scripts for data, features, models, API
│   ├── data/
│   ├── features/
│   ├── models/
│   └── api/
├── dashboard/               # Streamlit application
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
3. **Multi-Modal Fusion** → Vision + NLP for comprehensive diagnostics
4. **Experiment Tracking** → MLflow for metrics and versioning
5. **Deployment** → Dockerized with CI/CD via GitHub Actions
6. **Monitoring** → Drift detection, health scores, and anomaly tracking

---

## 🎮 Gamification & Dashboard Features

- **Multi-Plant Tracking**: Monitor multiple plants per user
- **Achievement Badges**: Earn rewards for maintaining perfect plants
- **Leaderboards**: Rankings for healthiest and most vulnerable plants
- **Time-Lapse Visualizations**: Track health evolution with interactive charts
- **Eco Metrics**: Environmental impact scores for each plant

---

## 📊 Datasets

The following datasets are used for training and validation:

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [Plant Disease Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)

---

## 🎯 Project Goals

Build a **comprehensive MLOps solution** for plant detection, diagnosis, and interactive monitoring that combines:
- **Computer Vision** for image analysis
- **Natural Language Processing** for contextual understanding
- **Gamification** for user engagement
- **Eco-Responsibility** for sustainable plant care

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Project Maintainers

This project is maintained by:

- **Asma Daab** - [LinkedIn]([https://www.linkedin.com/in/asma-daab](https://www.linkedin.com/in/asma-daab-b449051b6/))
- **Tesnime Ellabou** - [LinkedIn]([https://www.linkedin.com/in/tesnime-ellabou](https://www.linkedin.com/in/tesnime-ellabou-3170981b8/))

For questions, suggestions, or permission requests, please contact the maintainers via LinkedIn or open an issue.

---

This project and its source code are the exclusive property of Asma Daab and Tesnime Ellabou.

**You may NOT:**
- Use this code in any project (personal or commercial)
- Copy, modify, or distribute this code
- Deploy or host this application
- Use any part of this code without explicit written permission

**To request permission:** Contact the maintainers via LinkedIn

Copyright © 2024 Asma Daab & Tesnime Ellabou. All rights reserved.

---

**Made with 💚 for plant lovers and ML enthusiasts**
