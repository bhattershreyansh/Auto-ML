import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
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
