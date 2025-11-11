import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, Zap, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { toast } from "@/hooks/use-toast";
import { selectModel, trainModel } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface TrainingStepProps {
  filepath: string;
  analysisData: any;
  onNext: (metrics: any, modelPath: string) => void;
  onBack: () => void;
}

export function TrainingStep({ filepath, analysisData, onNext, onBack }: TrainingStepProps) {
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [suggestions, setSuggestions] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [testSize, setTestSize] = useState(0.2);
  const [tuneHyperparams, setTuneHyperparams] = useState<string>("false");
  const [cvFolds, setCvFolds] = useState(5);

  useEffect(() => {
    if (analysisData?.suggested_target) {
      setTargetColumn(analysisData.suggested_target);
      loadSuggestions(analysisData.suggested_target);
    }
  }, []);

  const loadSuggestions = async (target: string) => {
    setLoading(true);
    try {
      const data = await selectModel(filepath, target);
      setSuggestions(data);
      if (data.recommended_models?.[0]) {
        setSelectedModel(data.recommended_models[0]);
      }
    } catch (error: any) {
      toast({
        title: "Model selection failed",
        description: error.response?.data?.detail || "Failed to get model suggestions",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!selectedModel || !targetColumn) {
      toast({
        title: "Missing configuration",
        description: "Please select a model and target column",
        variant: "destructive",
      });
      return;
    }

    setTraining(true);
    try {
      const tuneValue = tuneHyperparams === "false" ? false : tuneHyperparams;
      const response = await trainModel(
        filepath,
        targetColumn,
        selectedModel,
        testSize,
        tuneValue,
        cvFolds,
        50
      );
      
      toast({
        title: "Training complete!",
        description: `Model trained successfully with ${selectedModel}`,
      });
      
      onNext(response.metrics, response.model_path);
    } catch (error: any) {
      toast({
        title: "Training failed",
        description: error.response?.data?.detail || "Failed to train model",
        variant: "destructive",
      });
    } finally {
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <Loader2 className="h-12 w-12 animate-spin text-primary" />
        <p className="text-muted-foreground">Analyzing model options...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold gradient-primary bg-clip-text text-transparent">
          Model Training
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Select your model and configure training parameters.
        </p>
      </div>

      <Tabs defaultValue="simple" className="w-full">
        <TabsList className="grid w-full max-w-md mx-auto grid-cols-2">
          <TabsTrigger value="simple">
            <Zap className="h-4 w-4 mr-2" />
            Quick Train
          </TabsTrigger>
          <TabsTrigger value="advanced">
            <Settings2 className="h-4 w-4 mr-2" />
            Advanced
          </TabsTrigger>
        </TabsList>

        <TabsContent value="simple" className="space-y-6 mt-6">
          <Card className="transition-smooth hover:shadow-lg">
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
              <CardDescription>Choose your target column and model</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="target">Target Column</Label>
                <Select value={targetColumn} onValueChange={(val) => {
                  setTargetColumn(val);
                  loadSuggestions(val);
                }}>
                  <SelectTrigger id="target">
                    <SelectValue placeholder="Select target column" />
                  </SelectTrigger>
                  <SelectContent>
                    {analysisData?.basic_statistics?.dtypes && Object.keys(analysisData.basic_statistics.dtypes).map((col: string) => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="model">Model</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger id="model">
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {suggestions?.recommended_models?.map((model: string) => (
                      <SelectItem key={model} value={model}>{model}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {suggestions && (
                <div className="p-4 rounded-lg bg-accent/10 border border-accent/20">
                  <p className="text-sm">
                    <span className="font-semibold">Detected Task:</span>{" "}
                    <span className="capitalize text-accent">{suggestions.task_type}</span>
                  </p>
                  <p className="text-sm mt-1">
                    <span className="font-semibold">Dataset Size:</span>{" "}
                    <span className="text-muted-foreground">{suggestions.data_size}</span>
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-6 mt-6">
          <Card className="transition-smooth hover:shadow-lg">
            <CardHeader>
              <CardTitle>Advanced Configuration</CardTitle>
              <CardDescription>Fine-tune your training parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="target-adv">Target Column</Label>
                  <Select value={targetColumn} onValueChange={(val) => {
                    setTargetColumn(val);
                    loadSuggestions(val);
                  }}>
                    <SelectTrigger id="target-adv">
                      <SelectValue placeholder="Select target column" />
                    </SelectTrigger>
                    <SelectContent>
                      {analysisData?.basic_statistics?.dtypes && Object.keys(analysisData.basic_statistics.dtypes).map((col: string) => (
                        <SelectItem key={col} value={col}>{col}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="model-adv">Model</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger id="model-adv">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      {suggestions?.recommended_models?.map((model: string) => (
                        <SelectItem key={model} value={model}>{model}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="test-size">Test Size (0.1 - 0.5)</Label>
                  <Input
                    id="test-size"
                    type="number"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    value={testSize}
                    onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="cv-folds">Cross-Validation Folds</Label>
                  <Input
                    id="cv-folds"
                    type="number"
                    min="2"
                    max="10"
                    value={cvFolds}
                    onChange={(e) => setCvFolds(parseInt(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="tune">Hyperparameter Tuning</Label>
                  <Select value={tuneHyperparams} onValueChange={setTuneHyperparams}>
                    <SelectTrigger id="tune">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="false">None</SelectItem>
                      <SelectItem value="grid">Grid Search</SelectItem>
                      <SelectItem value="optuna">Optuna (Bayesian)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Card className="p-6 bg-gradient-subtle border-primary/20">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h3 className="font-semibold text-lg">Ready to Train</h3>
            <p className="text-sm text-muted-foreground">
              Start training your {selectedModel || "selected"} model
            </p>
          </div>
          <Button
            size="lg"
            variant="gradient"
            onClick={handleTrain}
            disabled={training || !selectedModel || !targetColumn}
          >
            {training ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Training...
              </>
            ) : (
              "Start Training"
            )}
          </Button>
        </div>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack} disabled={training}>
          Back
        </Button>
      </div>
    </div>
  );
}
