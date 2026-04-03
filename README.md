# AutoPilot ML: Premium Open-Source AutoML 🚀

**AutoPilot ML** is a state-of-the-art, end-to-end AutoML platform designed to automate the entire machine learning lifecycle with a focus on **explainability**, **production-readiness**, and **premium UX**. It features a stunning "Metallic Obsidian" glassmorphism interface and a hardened MLOps backend.

![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-10b981.svg)
![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)
![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-purple.svg)
![Design](https://img.shields.io/badge/Design-Metallic--Obsidian-cyan.svg)

---

## ✨ Features

### 💎 **Premium "Metallic Obsidian" UI**
- **Glassmorphism Design**: High-end aesthetic with vibrant HSL tailored colors (Emerald, Cyan, Purple).
- **Interactive Wizard**: A seamless step-by-step guided workflow from raw data to production deployment.
- **Micro-Animations**: Smooth Framer Motion transitions and hover effects for an alive, interactive feel.

### 🔬 **Exploratory Intelligence**
- **XAI Diagnostics (SHAP)**: Deep feature importance insights using high-precision SHAP explainers.
- **Intelligent Analysis**: Automated EDA with correlation matrices, target skew diagnostics, and numeric variance audits.
- **Visual Telemetry**: Interactive Plotly charts themed perfectly for dark-mode environments.

### 🛡️ **Hardened MLOps Backend**
- **Target Leakage Guards**: Scans and purges features with >98% correlation to prevent "model cheating."
- **Class Imbalance Auto-Correction**: Injects `class_weight='balanced'` and `scale_pos_weight` to ensure minority classes are respected.
- **Defensive Cleaning**: Automatic dropping of columns with >50% missing data and IQR Winsorization for outlier clipping.

### 🚀 **Deployment Suite**
- **Interactive "What-If" Simulator**: Move feature sliders to get real-time prediction feedback from the trained model.
- **Champion Election**: Compare multiple architectures and elect a "Champion" for final synthesis and deployment.
- **Terminal Sandbox**: Auto-generates production-ready **FastAPI** code and **Dockerfiles** tailored exactly to your elected model.
- **One-Click Manifest**: Export your entire experiment and weights as a standardized deployment package.

---

## 🛠️ Technology Stack

### **Backend (Python Engine)**
- **FastAPI**: Asynchronous, high-performance API kernel.
- **Scikit-Learn**: The backbone for preprocessing pipelines and classical model architectures.
- **XGBoost**: Gradient boosting for elite predictive performance.
- **SHAP**: Kernel and Tree explainers for model transparency.
- **Optuna**: Bayesian optimization for hyperparameter tuning.
- **Neon/PostgreSQL**: Cloud-managed relational database for persistent session history.

### **Frontend (React Dashboard)**
- **React 18 + TypeScript**: Type-safe component architecture.
- **Tailwind CSS**: Utility-first styling for the Obsidian design system.
- **Shadcn/UI**: Modern, accessible UI primitives.
- **Plotly.js**: High-fidelity interactive data visualizations.
- **Framer Motion**: Smooth interface animations.

---

## 🚀 Getting Started

### 📦 Installation

1. **Clone the Project**
   ```bash
   git clone https://github.com/bhattershreyansh/Auto-ML.git
   cd Auto-ML
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend2/auto
   npm install
   ```

### 🎮 Running Locally

1. **Start the API (Port 8000)**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start the Dashboard (Port 3000)**
   ```bash
   cd frontend2/auto
   npm run dev
   ```

---

## 📡 Core API Lifecycle

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/upload` | `POST` | Ingest raw CSV data and initialize session. |
| `/analyze` | `POST` | Generate EDA stats and LLM-powered insights. |
| `/clean` | `POST` | Execute defensive cleaning strategies. |
| `/train` | `POST` | Execute model training with optional Optuna tuning. |
| `/predict` | `POST` | Real-time single-row inference for simulators. |
| `/download-assets` | `GET` | Export champion model and sanitized data as ZIP. |

---

## 🎯 The AutoPilot Workflow

1. **Ingest**: Upload any structured CSV dataset.
2. **Audit**: Review data quality reports and LLM-derived strategic advice.
3. **Refine**: Clean data and handle missingness/outliers automatically.
4. **Compare**: Train multiple models (XGB, RF, Logistic) and review metrics side-by-side.
5. **Elect**: Choose your Champion model and witness it retuned for production.
6. **Simulate**: Test the model's decision boundaries using interactive sliders.
7. **Deploy**: Copy the generated Docker/FastAPI code and build your production service.

---

**Built with Precision for the ML Community** 🥂
