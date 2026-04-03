import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Helper to fetch the auth token dynamically
let getClerkToken: (() => Promise<string | null>) | null = null;

export const setTokenFetcher = (fetcher: () => Promise<string | null>) => {
  getClerkToken = fetcher;
};

// Request interceptor for dynamic auth
api.interceptors.request.use(async (config) => {
  if (getClerkToken) {
    try {
      const token = await getClerkToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      } else {
        console.warn('API: Clerk token returned null');
      }
    } catch (err) {
      console.error('API: Failed to fetch Clerk token', err);
    }
  } else {
    console.warn('API: Token fetcher not registered yet');
  }
  return config;
});

export interface UploadResponse {
  filename: string;
  filepath: string;
  message: string;
}

export interface AnalyzeResponse {
  basic_statistics: {
    shape: [number, number];
    dtypes: Record<string, string>;
    nulls: Record<string, number>;
    unique_counts: Record<string, number>;
    sample_data: any[];
  };
  data_quality: {
    total_rows: number;
    total_columns: number;
    numeric_columns: number;
    categorical_columns: number;
    total_missing: number;
    missing_percentage: number;
    duplicate_rows: number;
    memory_usage_mb: number;
    columns_with_missing: Record<string, number>;
    constant_columns: string[];
    high_cardinality_columns: string[];
  };
  suggested_target: string;
  outliers: Record<string, number>;
  class_imbalance: any;
  correlations: any;
  llm_insights: {
    description: string;
    key_issues: string[];
    feature_suggestions: string[];
  };
  visualizations: Record<string, string>;
  summary_statistics: any;
}

export interface CleanResponse {
  cleaned_filepath: string;
  message: string;
}

export interface ModelSuggestion {
  recommended_models: string[];
  task_type: string;
  data_size: string;
  best_model?: {
    name: string;
    description: string;
  };
  other_options?: Array<{
    name: string;
    description: string;
  }>;
}

export interface TrainResponse {
  message: string;
  metrics: any;
  model_path: string;
  encoder_path?: string;
  model_filename: string;
  model_name: string;
  target_column: string;
}

export interface ModelComparisonResponse {
  message: string;
  comparison: {
    leaderboard: Array<{
      model_name: string;
      score: number;
      train_time: number;
      best_params?: any;
    }>;
    task_type: string;
  };
}

export const uploadFile = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const analyzeData = async (filepath: string): Promise<AnalyzeResponse> => {
  const response = await api.post<AnalyzeResponse>('/analyze', { filepath });
  return response.data;
};

export const cleanData = async (filepath: string): Promise<CleanResponse> => {
  const response = await api.post<CleanResponse>('/clean', { filepath });
  return response.data;
};

export const selectModel = async (filepath: string, targetColumn: string): Promise<ModelSuggestion> => {
  const response = await api.post<ModelSuggestion>('/select-model', {
    filepath,
    target_column: targetColumn,
  });
  return response.data;
};

export const trainModel = async (
  filepath: string,
  targetColumn: string,
  modelName: string,
  testSize: number = 0.2,
  tuneHyperparams: boolean | string = false,
  cvFolds: number = 5,
  nTrials: number = 50
): Promise<TrainResponse> => {
  const response = await api.post<TrainResponse>('/train', {
    filepath,
    target_column: targetColumn,
    model_name: modelName,
    test_size: testSize,
    tune_hyperparams: tuneHyperparams,
    cv_folds: cvFolds,
    n_trials: nTrials,
  });
  return response.data;
};

export const compareModels = async (
  filepath: string,
  targetColumn: string,
  modelNames?: string[],
  testSize: number = 0.2,
  tuneHyperparams: boolean = false,
  cvFolds: number = 3
): Promise<ModelComparisonResponse> => {
  const response = await api.post<ModelComparisonResponse>('/compare-models', {
    filepath,
    target_column: targetColumn,
    model_names: modelNames,
    test_size: testSize,
    tune_hyperparams: tuneHyperparams,
    cv_folds: cvFolds,
  });
  return response.data;
};

export interface PredictResponse {
  prediction: number;
  probabilities?: number[];
}

export const predictSingle = async (modelPath: string, features: Record<string, any>): Promise<PredictResponse> => {
  const response = await api.post<PredictResponse>('/predict', {
    model_path: modelPath,
    features,
  });
  return response.data;
};

export const downloadAssetsZip = async (modelPath: string) => {
  const response = await api.get('/download-assets', {
    params: { model_path: modelPath },
    responseType: 'blob',
  });
  
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/zip' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'autopilot-deployment.zip';
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  a.remove();
};
