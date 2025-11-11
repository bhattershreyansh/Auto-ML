# AutoML Platform Setup Guide

## Overview

This is a production-ready AutoML platform with a beautiful step-by-step wizard interface for machine learning workflows. Built with React, TypeScript, Tailwind CSS, and FastAPI.

## Features

✨ **6-Step Wizard Flow**
1. **Data Upload** - Drag-and-drop CSV file upload with validation
2. **Data Cleaning** - Automated data preprocessing and quality checks
3. **Exploratory Data Analysis** - Interactive visualizations and statistics
4. **Model Training** - Smart model selection with hyperparameter tuning
5. **Model Comparison** - Compare multiple algorithms side-by-side
6. **Results & Export** - Comprehensive metrics and export capabilities

🎨 **Beautiful Design**
- Purple-blue gradient theme inspired by Vertex AI and Retool
- Smooth animations and transitions
- Responsive layout for all screen sizes
- Dark mode support (inherit from system)
- Semantic design tokens for consistency

🚀 **Advanced Features**
- Real-time training progress
- Hyperparameter tuning (Grid Search & Optuna)
- Cross-validation support
- Model leaderboards with rankings
- JSON export for results
- Contextual help and tooltips
- Error handling with toast notifications

## Prerequisites

### Frontend
- Node.js 18+ and npm
- Modern web browser

### Backend
- Python 3.8+
- FastAPI
- Your custom AutoML services (see `backend-reference/main.py`)

## Installation

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The app will be available at `http://localhost:5173`

### Backend Setup

The backend API is in `backend-reference/main.py`. You'll need to:

1. Install Python dependencies:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

2. Set up your service modules:
- `services/analyzer.py` - Data analysis
- `services/cleaner.py` - Data cleaning
- `services/model_selector.py` - Model selection
- `services/trainer.py` - Model training
- `services/tester.py` - Model evaluation
- `services/model_comparator.py` - Model comparison

3. Run the FastAPI server:
```bash
cd backend-reference
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /upload
Upload CSV file for analysis
- **Body**: `multipart/form-data` with CSV file
- **Returns**: `{ filename, filepath, message }`

### POST /analyze
Analyze uploaded dataset
- **Body**: `{ filepath: string }`
- **Returns**: Dataset analysis with shape, dtypes, missing values, target suggestions

### POST /clean
Clean the dataset
- **Body**: `{ filepath: string }`
- **Returns**: `{ cleaned_filepath: string, message: string }`

### POST /select-model
Get model suggestions
- **Body**: `{ filepath: string, target_column: string }`
- **Returns**: `{ recommended_models: string[], task_type: string, data_size: string }`

### POST /train
Train selected model
- **Body**: 
  ```json
  {
    "filepath": "string",
    "target_column": "string",
    "model_name": "string",
    "test_size": 0.2,
    "tune_hyperparams": false | "grid" | "optuna",
    "cv_folds": 5,
    "n_trials": 50
  }
  ```
- **Returns**: Training metrics, model path, encoder path

### POST /compare-models
Compare multiple models
- **Body**: 
  ```json
  {
    "filepath": "string",
    "target_column": "string",
    "model_names": ["string"],
    "test_size": 0.2,
    "tune_hyperparams": false,
    "cv_folds": 3
  }
  ```
- **Returns**: Model leaderboard with scores and training times

## Configuration

### API Base URL

Update the API base URL in `src/lib/api.ts`:

```typescript
const API_BASE_URL = 'http://localhost:8000'; // Change for production
```

### CORS Configuration

Update CORS settings in `backend-reference/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Design System

### Color Palette

The app uses a semantic color system defined in `src/index.css`:

- **Primary**: Purple gradient (`#8B5CF6` to `#C084FC`)
- **Accent**: Emerald green (`#10B981`)
- **Background**: Light gray (`#F8F9FB`)
- **Foreground**: Dark gray (`#1F2937`)

### Custom CSS Utilities

```css
.gradient-primary - Purple-blue gradient background
.gradient-accent - Green gradient background
.gradient-subtle - Subtle gray gradient
.shadow-glow - Glowing shadow effect
.transition-smooth - Smooth transitions
```

### Button Variants

- `variant="default"` - Standard primary button
- `variant="gradient"` - Gradient purple button with glow
- `variant="accent"` - Gradient green button
- `variant="outline"` - Outlined button
- `variant="ghost"` - Transparent button
- `variant="secondary"` - Secondary gray button

## Usage Flow

1. **Start**: User lands on upload page
2. **Upload CSV**: Drag-and-drop or click to upload dataset
3. **Data Preview**: System analyzes and shows data quality metrics
4. **Cleaning**: Automated cleaning with summary of changes
5. **EDA**: View distributions, correlations, and statistics
6. **Configure Training**: Select target column, model, and hyperparameters
7. **Train Model**: Real-time training progress with metrics
8. **Compare Models**: Optional comparison across multiple algorithms
9. **Results**: View comprehensive metrics and export

## Development Tips

### Adding New Steps

1. Create step component in `src/components/steps/`
2. Add step to wizard in `src/pages/Index.tsx`
3. Update steps array with new step info
4. Implement navigation handlers

### Customizing Design

All design tokens are in `src/index.css`. Update:
- Color variables (use HSL format only)
- Gradients
- Shadows
- Transitions

### Adding API Endpoints

1. Add function to `src/lib/api.ts`
2. Define TypeScript interfaces for request/response
3. Use in step components with proper error handling

## Troubleshooting

### CORS Errors
- Ensure backend CORS is configured for frontend URL
- Check API base URL in `src/lib/api.ts`

### Build Errors
- Run `npm install` to ensure all dependencies are installed
- Check TypeScript errors with `npm run build`

### API Connection Issues
- Verify backend is running on correct port
- Check network requests in browser DevTools
- Validate API endpoint URLs

## Deployment

### Frontend (Vercel/Netlify)
```bash
npm run build
# Deploy dist/ folder
```

### Backend (Docker)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

This is a demo/template project for AutoML workflows.

## Support

For issues or questions, refer to:
- React documentation: https://react.dev
- Tailwind CSS: https://tailwindcss.com
- FastAPI: https://fastapi.tiangolo.com
- shadcn/ui: https://ui.shadcn.com
