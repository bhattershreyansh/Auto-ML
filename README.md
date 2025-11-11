# AutoML Platform 🚀

A comprehensive end-to-end AutoML platform that automates the entire machine learning workflow from data analysis to model deployment. Built with FastAPI backend and React frontend, featuring intelligent model selection, automated preprocessing, and advanced hyperparameter tuning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-3178c6.svg)

## ✨ Features

### 🔍 **Intelligent Data Analysis**
- Automated Exploratory Data Analysis (EDA) with comprehensive statistics
- LLM-powered dataset insights using Groq API
- Interactive visualizations (correlation heatmaps, distributions, scatter plots)
- Outlier detection and class imbalance analysis
- Data quality reports with missing value analysis

### 🧹 **Automated Data Cleaning**
- Smart missing value imputation (median for numeric, mode for categorical)
- Automatic data type detection and handling
- Duplicate row removal
- Data validation and quality checks

### 🤖 **Smart Model Selection**
- LLM-assisted model recommendation based on dataset characteristics
- Support for 7+ machine learning algorithms:
  - **Classification**: RandomForest, GradientBoosting, XGBoost, LogisticRegression
  - **Regression**: RandomForest, GradientBoosting, XGBoost
- Automatic task type detection (classification vs regression)

### 🎯 **Advanced Model Training**
- Automated preprocessing pipeline with ColumnTransformer
- Feature scaling and encoding
- Hyperparameter tuning with two methods:
  - **Grid Search**: Exhaustive search over parameter grid
  - **Optuna**: Bayesian optimization for efficient tuning
- Cross-validation support
- Comprehensive evaluation metrics

### 📊 **Model Comparison**
- Side-by-side comparison of multiple models
- Performance metrics visualization
- Best model recommendation based on evaluation scores

### 🎨 **Interactive Visualizations**
- Plotly-powered interactive charts
- Real-time performance metrics
- Feature importance analysis
- Model comparison dashboards

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Optuna** - Hyperparameter optimization
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Groq** - LLM API for intelligent recommendations

### Frontend
- **React 18** - UI library
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - High-quality component library
- **Plotly.js** - Interactive charting
- **React Query** - Data fetching and caching
- **React Router** - Client-side routing

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 18+ and npm/yarn
- Groq API key (optional, for LLM features)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bhattershreyansh/Auto-ML.git
cd Auto-ML
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the `backend` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Note**: The Groq API key is optional. The platform will work without it, but LLM-powered features will be disabled.

### 4. Frontend Setup

```bash
cd frontend2/auto

# Install dependencies
npm install

# Or with yarn
yarn install
```

## 🎮 Usage

### Starting the Backend Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

API documentation (Swagger UI) will be available at `http://localhost:8000/docs`

### Starting the Frontend Development Server

```bash
cd frontend2/auto
npm run dev
```

The frontend will be available at `http://localhost:8080` (or the port shown in terminal)

## 📡 API Endpoints

### File Management
- `POST /upload` - Upload CSV file for analysis
- `GET /health` - Health check endpoint

### Data Analysis
- `POST /analyze` - Perform comprehensive dataset analysis
- `POST /clean` - Clean and preprocess dataset

### Model Operations
- `POST /select-model` - Get AI-powered model recommendations
- `POST /train` - Train a machine learning model
- `POST /evaluate` - Evaluate trained model on test data
- `POST /predict` - Make predictions with trained model
- `POST /compare-models` - Compare multiple models side-by-side

## 📁 Project Structure

```
Auto-ML/
├── backend/
│   ├── main.py                 # FastAPI application and routes
│   ├── services/
│   │   ├── analyzer.py         # Dataset analysis and EDA
│   │   ├── cleaner.py          # Data cleaning and preprocessing
│   │   ├── model_selector.py   # LLM-powered model selection
│   │   ├── trainer.py          # Model training and hyperparameter tuning
│   │   ├── tester.py           # Model evaluation
│   │   └── model_comparator.py # Model comparison utilities
│   └── uploads/                # Uploaded datasets and trained models
│
├── frontend2/
│   └── auto/
│       ├── src/
│       │   ├── components/    # React components
│       │   │   ├── steps/      # Wizard step components
│       │   │   ├── charts/     # Visualization components
│       │   │   └── ui/         # shadcn/ui components
│       │   ├── lib/            # Utilities and API client
│       │   └── pages/           # Page components
│       └── package.json
│
└── README.md
```

## 🔧 Configuration

### Hyperparameter Tuning Options

When training a model, you can specify:

- `tune_hyperparams`: `false` (default), `'grid'`, or `'optuna'`
- `cv_folds`: Number of cross-validation folds (default: 5)
- `n_trials`: Number of Optuna optimization trials (default: 50)

### Supported Models

**Classification:**
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `XGBClassifier`
- `LogisticRegression`

**Regression:**
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `XGBRegressor`

## 🎯 Workflow

1. **Upload Dataset** - Upload your CSV file through the web interface
2. **Analyze Data** - Get comprehensive insights about your dataset
3. **Clean Data** - Automatically handle missing values and data quality issues
4. **Select Model** - Get AI-powered recommendations for the best model
5. **Train Model** - Train with optional hyperparameter tuning
6. **Evaluate** - Assess model performance on test data
7. **Compare** - Compare multiple models to find the best one
8. **Predict** - Use trained models to make predictions

## 🧪 Example Usage

### Training a Model via API

```python
import requests

# Upload dataset
with open('dataset.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
    filepath = response.json()['filepath']

# Analyze dataset
analyze_response = requests.post('http://localhost:8000/analyze', 
                                json={'filepath': filepath})

# Train model
train_response = requests.post('http://localhost:8000/train', json={
    'filepath': filepath,
    'target_column': 'target',
    'model_name': 'XGBClassifier',
    'tune_hyperparams': 'optuna',
    'n_trials': 50
})
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- scikit-learn team for the excellent ML library
- FastAPI for the amazing web framework
- React and the open-source community
- Groq for LLM API access

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the ML community**

