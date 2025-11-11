import { Download, CheckCircle2, BarChart3, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

interface ResultsStepProps {
  metrics: any;
  comparisonData?: any;
  onRestart: () => void;
}

export function ResultsStep({ metrics, comparisonData, onRestart }: ResultsStepProps) {
  const handleExport = () => {
    const results = {
      training_metrics: metrics,
      comparison: comparisonData,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `automl-results-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="text-center space-y-2">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent/10 mb-4">
          <CheckCircle2 className="h-8 w-8 text-accent" />
        </div>
        <h2 className="text-3xl font-bold gradient-primary bg-clip-text text-transparent">
          AutoML Complete!
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Your models have been trained and evaluated. Review the results below.
        </p>
      </div>

      <Card className="transition-smooth hover:shadow-lg border-accent/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Training Metrics
          </CardTitle>
          <CardDescription>Performance metrics from your trained model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            {Object.entries(metrics || {})
              .filter(([key, value]) => 
                typeof value === 'number' && 
                !['support'].includes(key) && // Exclude support counts
                !key.includes('avg') // Exclude avg metrics for now, show them separately
              )
              .map(([key, value]: [string, any]) => (
              <div key={key} className="p-4 rounded-lg bg-gradient-subtle border border-border">
                <p className="text-sm text-muted-foreground capitalize mb-1">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-2xl font-bold text-primary">
                  {value.toFixed(4)}
                </p>
              </div>
            ))}
            
            {/* Show macro averages from classification report if available */}
            {metrics?.['macro avg'] && typeof metrics['macro avg'] === 'object' && (
              <>
                {['precision', 'recall', 'f1-score'].map((metric) => (
                  metrics['macro avg'][metric] && (
                    <div key={`macro_${metric}`} className="p-4 rounded-lg bg-gradient-subtle border border-border">
                      <p className="text-sm text-muted-foreground capitalize mb-1">
                        {metric.replace('-', ' ')} (Macro)
                      </p>
                      <p className="text-2xl font-bold text-primary">
                        {metrics['macro avg'][metric].toFixed(4)}
                      </p>
                    </div>
                  )
                ))}
              </>
            )}
          </div>
          
          {/* Show detailed metrics for classification */}
          {metrics?.classification_report && typeof metrics.classification_report === 'object' && (
            <div className="mt-6">
              <h4 className="text-sm font-semibold mb-3">Detailed Classification Metrics</h4>
              <div className="grid md:grid-cols-3 gap-4">
                {['precision', 'recall', 'f1-score'].map((metricType) => (
                  <div key={metricType} className="space-y-2">
                    <p className="text-sm font-medium capitalize">{metricType.replace('-', ' ')}</p>
                    {Object.entries(metrics.classification_report)
                      .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                      .map(([className, classMetrics]: [string, any]) => (
                        classMetrics && typeof classMetrics === 'object' && classMetrics[metricType] && (
                          <div key={className} className="flex justify-between text-sm">
                            <span className="text-muted-foreground">Class {className}:</span>
                            <span className="font-medium">{classMetrics[metricType].toFixed(3)}</span>
                          </div>
                        )
                      ))}
                    {/* Show averages */}
                    {metrics.classification_report['macro avg'] && (
                      <div className="flex justify-between text-sm border-t pt-1">
                        <span className="text-muted-foreground">Macro Avg:</span>
                        <span className="font-semibold">
                          {metrics.classification_report['macro avg'][metricType]?.toFixed(3)}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {comparisonData?.leaderboard && (
        <>
          <Separator />
          
          <Card className="transition-smooth hover:shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="h-5 w-5 text-accent" />
                Model Comparison Summary
              </CardTitle>
              <CardDescription>Top performing models from the comparison</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {comparisonData.leaderboard.slice(0, 3).map((model: any, idx: number) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-4 rounded-lg bg-muted/50 hover:bg-muted transition-smooth"
                >
                  <div className="flex items-center gap-4">
                    <Badge className={idx === 0 ? "gradient-accent text-white" : "bg-secondary"}>
                      #{idx + 1}
                    </Badge>
                    <div>
                      <p className="font-semibold">{model.model_name}</p>
                      <p className="text-sm text-muted-foreground">
                        Training time: {model.train_time?.toFixed(2)}s
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-primary">
                      {model.score?.toFixed(4)}
                    </p>
                    <p className="text-xs text-muted-foreground">score</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      )}

      <Card className="p-6 bg-gradient-subtle border-primary/20">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-center md:text-left">
            <h3 className="font-semibold text-lg mb-1">Export Your Results</h3>
            <p className="text-sm text-muted-foreground">
              Download a comprehensive report of all metrics and comparisons
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              size="lg"
              onClick={handleExport}
            >
              <Download className="mr-2 h-4 w-4" />
              Export JSON
            </Button>
            <Button
              size="lg"
              variant="gradient"
              onClick={onRestart}
            >
              Start New Project
            </Button>
          </div>
        </div>
      </Card>

      <div className="text-center pt-6">
        <p className="text-sm text-muted-foreground">
          Thank you for using AutoML Platform! Your trained models are ready for deployment.
        </p>
      </div>
    </div>
  );
}
