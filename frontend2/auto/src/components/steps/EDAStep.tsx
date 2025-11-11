import { BarChart3, TrendingUp, Activity, PieChart } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PlotlyChart } from "@/components/charts/PlotlyChart";

interface EDAStepProps {
  analysisData: any;
  onNext: () => void;
  onBack: () => void;
}

export function EDAStep({ analysisData, onNext, onBack }: EDAStepProps) {
  console.log('EDAStep - analysisData:', analysisData);
  console.log('EDAStep - visualizations:', analysisData?.visualizations);
  console.log('EDAStep - visualization keys:', analysisData?.visualizations ? Object.keys(analysisData.visualizations) : 'No visualizations');
  
  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold gradient-primary bg-clip-text text-transparent">
          Exploratory Data Analysis
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Understand your data through visualizations and statistical insights.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card className="transition-smooth hover:shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Column Types
            </CardTitle>
            <CardDescription>Data types detected in your dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {analysisData?.basic_statistics?.dtypes && Object.entries(analysisData.basic_statistics.dtypes).slice(0, 10).map(([col, dtype]: [string, any]) => (
                <div key={col} className="flex justify-between items-center py-2 border-b last:border-0">
                  <span className="text-sm font-medium truncate max-w-[200px]">{col}</span>
                  <Badge variant="secondary">{dtype}</Badge>
                </div>
              ))}
              {analysisData?.basic_statistics?.dtypes && Object.keys(analysisData.basic_statistics.dtypes).length > 10 && (
                <p className="text-xs text-muted-foreground text-center pt-2">
                  + {Object.keys(analysisData.basic_statistics.dtypes).length - 10} more columns
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="transition-smooth hover:shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-accent" />
              Quick Stats
            </CardTitle>
            <CardDescription>Key metrics about your dataset</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">Total Rows</p>
                <p className="text-2xl font-bold text-primary mt-1">
                  {analysisData?.basic_statistics?.shape?.[0]?.toLocaleString() || 0}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">Total Columns</p>
                <p className="text-2xl font-bold text-primary mt-1">
                  {analysisData?.basic_statistics?.shape?.[1] || 0}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50 col-span-2">
                <p className="text-sm text-muted-foreground">Target Column</p>
                <p className="text-xl font-bold text-accent mt-1 capitalize">
                  {analysisData?.suggested_target || "Unknown"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Visualizations */}
      {analysisData?.visualizations && Object.keys(analysisData.visualizations).length > 0 ? (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Data Visualizations</h3>
          <Tabs defaultValue="distributions" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="distributions" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Distributions
              </TabsTrigger>
              <TabsTrigger value="correlations" className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Correlations
              </TabsTrigger>
              <TabsTrigger value="relationships" className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Relationships
              </TabsTrigger>
              <TabsTrigger value="quality" className="flex items-center gap-2">
                <PieChart className="h-4 w-4" />
                Data Quality
              </TabsTrigger>
            </TabsList>

            <TabsContent value="distributions" className="mt-6">
              <div className="grid md:grid-cols-2 gap-6">
                {analysisData.visualizations.target_distribution && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Target Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <PlotlyChart data={analysisData.visualizations.target_distribution} />
                    </CardContent>
                  </Card>
                )}
                {analysisData.visualizations.numeric_distributions && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Numeric Features</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <PlotlyChart data={analysisData.visualizations.numeric_distributions} />
                    </CardContent>
                  </Card>
                )}
                {Object.entries(analysisData.visualizations)
                  .filter(([key]) => key.startsWith('categorical_'))
                  .slice(0, 2)
                  .map(([key, data]) => (
                    <Card key={key}>
                      <CardHeader>
                        <CardTitle className="text-base">
                          {key.replace('categorical_', '').replace('_', ' ')}
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <PlotlyChart data={data as string} />
                      </CardContent>
                    </Card>
                  ))}
              </div>
            </TabsContent>

            <TabsContent value="correlations" className="mt-6">
              <div className="grid gap-6">
                {analysisData.visualizations.correlation_heatmap && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Feature Correlation Heatmap</CardTitle>
                      <CardDescription>
                        Correlation between numeric features (-1 to 1)
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PlotlyChart data={analysisData.visualizations.correlation_heatmap} />
                    </CardContent>
                  </Card>
                )}
              </div>
            </TabsContent>

            <TabsContent value="relationships" className="mt-6">
              <div className="grid gap-6">
                {analysisData.visualizations.scatter_matrix && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Feature Relationships</CardTitle>
                      <CardDescription>
                        Pairwise relationships between numeric features
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PlotlyChart data={analysisData.visualizations.scatter_matrix} />
                    </CardContent>
                  </Card>
                )}
                {analysisData.visualizations.target_vs_feature && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Target vs Features</CardTitle>
                      <CardDescription>
                        How features relate to the target variable
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PlotlyChart data={analysisData.visualizations.target_vs_feature} />
                    </CardContent>
                  </Card>
                )}
              </div>
            </TabsContent>

            <TabsContent value="quality" className="mt-6">
              <div className="grid md:grid-cols-2 gap-6">
                {analysisData.visualizations.missing_values && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Missing Values</CardTitle>
                      <CardDescription>
                        Columns with missing data
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <PlotlyChart data={analysisData.visualizations.missing_values} />
                    </CardContent>
                  </Card>
                )}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Data Quality Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Missing Values:</span>
                        <span className="font-medium">
                          {analysisData.data_quality?.missing_percentage || 0}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Duplicate Rows:</span>
                        <span className="font-medium">
                          {analysisData.data_quality?.duplicate_rows || 0}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Memory Usage:</span>
                        <span className="font-medium">
                          {analysisData.data_quality?.memory_usage_mb || 0} MB
                        </span>
                      </div>
                      {analysisData.data_quality?.constant_columns?.length > 0 && (
                        <div className="pt-2 border-t">
                          <p className="text-sm text-muted-foreground mb-1">Constant Columns:</p>
                          <div className="flex flex-wrap gap-1">
                            {analysisData.data_quality.constant_columns.map((col: string) => (
                              <Badge key={col} variant="outline" className="text-xs">
                                {col}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </Card>
      ) : (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Data Visualizations</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="aspect-video rounded-lg bg-gradient-subtle border border-border flex items-center justify-center">
              <div className="text-center space-y-2">
                <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  No visualization data available
                </p>
              </div>
            </div>
            <div className="aspect-video rounded-lg bg-gradient-subtle border border-border flex items-center justify-center">
              <div className="text-center space-y-2">
                <TrendingUp className="h-12 w-12 mx-auto text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Run analysis to generate charts
                </p>
              </div>
            </div>
          </div>
        </Card>
      )}

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
        <Button
          size="lg"
          variant="gradient"
          onClick={onNext}
        >
          Next: Train Models
        </Button>
      </div>
    </div>
  );
}
