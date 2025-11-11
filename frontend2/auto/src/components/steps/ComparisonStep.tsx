import { useState } from "react";
import { Trophy, Clock, Zap, Loader2, BarChart3, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/hooks/use-toast";
import { compareModels } from "@/lib/api";
import { CorrelationMatrix } from "@/components/charts/CorrelationMatrix";
import { PerformanceChart } from "@/components/charts/PerformanceChart";

interface ComparisonStepProps {
  filepath: string;
  targetColumn: string;
  currentMetrics: any;
  onNext: (comparisonData: any) => void;
  onBack: () => void;
}

export function ComparisonStep({ filepath, targetColumn, currentMetrics, onNext, onBack }: ComparisonStepProps) {
  const [comparing, setComparing] = useState(false);
  const [comparisonData, setComparisonData] = useState<any>(null);

  const handleCompare = async () => {
    setComparing(true);
    try {
      const response = await compareModels(filepath, targetColumn, undefined, 0.2, false, 3);
      console.log('Comparison response:', response);
      console.log('Comparison data:', response.comparison);
      console.log('Leaderboard:', response.comparison?.leaderboard);
      setComparisonData(response.comparison);
      toast({
        title: "Comparison complete!",
        description: "All models have been evaluated",
      });
    } catch (error: any) {
      console.error('Comparison error:', error);
      toast({
        title: "Comparison failed",
        description: error.response?.data?.detail || "Failed to compare models",
        variant: "destructive",
      });
    } finally {
      setComparing(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold gradient-primary bg-clip-text text-transparent">
          Model Comparison
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Compare multiple models to find the best performer for your dataset.
        </p>
      </div>

      {currentMetrics && (
        <Card className="transition-smooth hover:shadow-lg border-accent/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trophy className="h-5 w-5 text-accent" />
              Current Model Performance
            </CardTitle>
            <CardDescription>Results from your trained model</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4">
              {Object.entries(currentMetrics)
                .filter(([key, value]) => 
                  typeof value === 'number' && 
                  !['support'].includes(key) && // Exclude support counts
                  !key.includes('avg') // Exclude avg metrics for now, show them separately
                )
                .slice(0, 6)
                .map(([key, value]: [string, any]) => (
                <div key={key} className="p-4 rounded-lg bg-muted/50">
                  <p className="text-sm text-muted-foreground capitalize">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-xl font-bold text-primary mt-1">
                    {value.toFixed(4)}
                  </p>
                </div>
              ))}
              
              {/* Show macro averages from classification report if available */}
              {currentMetrics['macro avg'] && typeof currentMetrics['macro avg'] === 'object' && (
                <>
                  {['precision', 'recall', 'f1-score'].map((metric) => (
                    currentMetrics['macro avg'][metric] && (
                      <div key={`macro_${metric}`} className="p-4 rounded-lg bg-muted/50">
                        <p className="text-sm text-muted-foreground capitalize">
                          {metric.replace('-', ' ')} (Macro)
                        </p>
                        <p className="text-xl font-bold text-primary mt-1">
                          {currentMetrics['macro avg'][metric].toFixed(4)}
                        </p>
                      </div>
                    )
                  ))}
                </>
              )}
            </div>
            
            {/* Show detailed metrics for classification */}
            {currentMetrics.classification_report && typeof currentMetrics.classification_report === 'object' && (
              <div className="mt-6">
                <h4 className="text-sm font-semibold mb-3">Detailed Classification Metrics</h4>
                <div className="grid md:grid-cols-3 gap-4">
                  {['precision', 'recall', 'f1-score'].map((metricType) => (
                    <div key={metricType} className="space-y-2">
                      <p className="text-sm font-medium capitalize">{metricType.replace('-', ' ')}</p>
                      {Object.entries(currentMetrics.classification_report)
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
                      {currentMetrics.classification_report['macro avg'] && (
                        <div className="flex justify-between text-sm border-t pt-1">
                          <span className="text-muted-foreground">Macro Avg:</span>
                          <span className="font-semibold">
                            {currentMetrics.classification_report['macro avg'][metricType]?.toFixed(3)}
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
      )}

      {!comparisonData ? (
        <Card className="p-8 text-center">
          <div className="max-w-md mx-auto space-y-4">
            <Zap className="h-16 w-16 mx-auto text-primary" />
            <h3 className="text-xl font-semibold">Compare All Models</h3>
            <p className="text-muted-foreground">
              Run a comprehensive comparison across all available models to identify
              the best algorithm for your specific use case.
            </p>
            <Button
              size="lg"
              variant="gradient"
              onClick={handleCompare}
              disabled={comparing}
            >
              {comparing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Comparing Models...
                </>
              ) : (
                "Start Comparison"
              )}
            </Button>
          </div>
        </Card>
      ) : (
        <Tabs defaultValue="leaderboard" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="leaderboard" className="flex items-center gap-2">
              <Trophy className="h-4 w-4" />
              Leaderboard
            </TabsTrigger>
            <TabsTrigger value="charts" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Performance Charts
            </TabsTrigger>
            <TabsTrigger value="correlation" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Correlation Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="leaderboard">
            <Card className="transition-smooth hover:shadow-lg">
              <CardHeader>
                <CardTitle>Model Leaderboard</CardTitle>
                <CardDescription>
                  Ranked by performance score ({comparisonData.task_type})
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead>Score</TableHead>
                      <TableHead>Training Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {comparisonData.leaderboard && comparisonData.leaderboard.length > 0 ? (
                      comparisonData.leaderboard.map((model: any, idx: number) => (
                        <TableRow key={idx}>
                          <TableCell>
                            {idx === 0 ? (
                              <Badge className="gradient-accent text-white">
                                <Trophy className="h-3 w-3 mr-1" />
                                #{idx + 1}
                              </Badge>
                            ) : (
                              <span className="text-muted-foreground">#{idx + 1}</span>
                            )}
                          </TableCell>
                          <TableCell className="font-medium">{model.model_name}</TableCell>
                          <TableCell>
                            <span className="font-semibold text-primary">
                              {model.score?.toFixed(4) || 'N/A'}
                            </span>
                          </TableCell>
                          <TableCell className="flex items-center gap-2 text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            {model.train_time?.toFixed(2) || 'N/A'}s
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center text-muted-foreground">
                          No comparison data available
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="charts">
            {comparisonData.chart_data && comparisonData.chart_data.length > 0 ? (
              <PerformanceChart 
                data={comparisonData.chart_data} 
                taskType={comparisonData.task_type}
              />
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <p className="text-muted-foreground">No chart data available</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="correlation">
            {comparisonData.correlation_data && comparisonData.correlation_data.correlation_matrix ? (
              <CorrelationMatrix 
                data={comparisonData.correlation_data.correlation_matrix}
                columns={comparisonData.correlation_data.columns}
                targetCorrelations={comparisonData.correlation_data.target_correlations}
              />
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <p className="text-muted-foreground">No correlation data available</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      )}

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack} disabled={comparing}>
          Back
        </Button>
        <Button
          size="lg"
          variant="gradient"
          onClick={() => onNext(comparisonData)}
          disabled={!comparisonData}
        >
          Next: View Results
        </Button>
      </div>
    </div>
  );
}
