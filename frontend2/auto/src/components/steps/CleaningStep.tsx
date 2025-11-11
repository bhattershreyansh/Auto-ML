import { useState, useEffect } from "react";
import { Loader2, Sparkles, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "@/hooks/use-toast";
import { analyzeData, cleanData } from "@/lib/api";

interface CleaningStepProps {
  filepath: string;
  onNext: (cleanedFilepath: string, analysisData: any) => void;
  onBack: () => void;
}

export function CleaningStep({ filepath, onNext, onBack }: CleaningStepProps) {
  const [loading, setLoading] = useState(true);
  const [cleaning, setCleaning] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [cleaned, setCleaned] = useState(false);

  useEffect(() => {
    loadAnalysis();
  }, [filepath]);

  const loadAnalysis = async () => {
    setLoading(true);
    try {
      const data = await analyzeData(filepath);
      console.log('Analysis data received:', data);
      console.log('Visualizations in data:', data.visualizations);
      console.log('Visualization keys:', data.visualizations ? Object.keys(data.visualizations) : 'No visualizations');
      setAnalysisData(data);
    } catch (error: any) {
      console.error('Analysis error:', error);
      toast({
        title: "Analysis failed",
        description: error.response?.data?.detail || "Failed to analyze data",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleClean = async () => {
    setCleaning(true);
    try {
      const response = await cleanData(filepath);
      setCleaned(true);
      toast({
        title: "Data cleaned!",
        description: "Your data has been processed and cleaned",
      });
      
      // Re-analyze cleaned data
      setTimeout(() => {
        onNext(response.cleaned_filepath, analysisData);
      }, 1000);
    } catch (error: any) {
      toast({
        title: "Cleaning failed",
        description: error.response?.data?.detail || "Failed to clean data",
        variant: "destructive",
      });
    } finally {
      setCleaning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <Loader2 className="h-12 w-12 animate-spin text-primary" />
        <p className="text-muted-foreground">Analyzing your dataset...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold gradient-primary bg-clip-text text-transparent">
          Data Cleaning & Preparation
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Review your data quality and let us handle missing values, outliers, and formatting.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card className="transition-smooth hover:shadow-lg">
          <CardHeader>
            <CardTitle>Dataset Overview</CardTitle>
            <CardDescription>Basic statistics about your data</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Rows</p>
                <p className="text-2xl font-bold text-primary">{analysisData?.basic_statistics?.shape?.[0] || 0}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Columns</p>
                <p className="text-2xl font-bold text-primary">{analysisData?.basic_statistics?.shape?.[1] || 0}</p>
              </div>
            </div>
            
            <div className="space-y-2">
              <p className="text-sm font-medium">Target Column</p>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium">
                {analysisData?.suggested_target || "Unknown"}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="transition-smooth hover:shadow-lg">
          <CardHeader>
            <CardTitle>Data Quality</CardTitle>
            <CardDescription>Missing values and data types</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-sm font-medium">Missing Values</p>
              {analysisData?.basic_statistics?.nulls && Object.keys(analysisData.basic_statistics.nulls).length > 0 ? (
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {Object.entries(analysisData.basic_statistics.nulls).map(([col, count]: [string, any]) => (
                    count > 0 && (
                      <div key={col} className="flex justify-between text-sm">
                        <span className="text-muted-foreground">{col}</span>
                        <span className="font-medium text-destructive">{count}</span>
                      </div>
                    )
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-accent" />
                  No missing values detected
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {analysisData?.suggested_target && (
        <Alert className="border-accent/50 bg-accent/5">
          <Sparkles className="h-4 w-4 text-accent" />
          <AlertDescription>
            <strong>Suggested target column:</strong>{" "}
            {analysisData.suggested_target}
          </AlertDescription>
        </Alert>
      )}

      <Card className="p-6 bg-gradient-subtle border-primary/20">
        <div className="flex items-start gap-4">
          <div className="flex-1 space-y-2">
            <h3 className="font-semibold text-lg">Ready to Clean?</h3>
            <p className="text-sm text-muted-foreground">
              Our automated cleaning process will handle missing values, remove duplicates,
              and standardize formats to prepare your data for model training.
            </p>
          </div>
          <Button
            size="lg"
            variant="gradient"
            onClick={handleClean}
            disabled={cleaning || cleaned}
          >
            {cleaning ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Cleaning...
              </>
            ) : cleaned ? (
              <>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Cleaned!
              </>
            ) : (
              "Clean Data"
            )}
          </Button>
        </div>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack} disabled={cleaning}>
          Back
        </Button>
        <Button
          size="lg"
          variant="gradient"
          onClick={() => cleaned && onNext(filepath, analysisData)}
          disabled={!cleaned}
        >
          Next: Exploratory Analysis
        </Button>
      </div>
    </div>
  );
}
