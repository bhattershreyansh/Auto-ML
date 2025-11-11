import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ChartData {
  model: string;
  score: number;
  train_time: number;
}

interface PerformanceChartProps {
  data: ChartData[];
  taskType: string;
}

export function PerformanceChart({ data, taskType }: PerformanceChartProps) {
  const metricName = taskType === 'classification' ? 'Accuracy' : 'R² Score';

  return (
    <div className="space-y-6">
      {/* Performance Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Comparison</CardTitle>
          <CardDescription>
            {metricName} scores for each model
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="model" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  interval={0}
                />
                <YAxis 
                  label={{ value: metricName, angle: -90, position: 'insideLeft' }}
                  domain={['dataMin - 0.05', 'dataMax + 0.05']}
                />
                <Tooltip 
                  formatter={(value: number) => [value.toFixed(4), metricName]}
                  labelFormatter={(label) => `Model: ${label}`}
                />
                <Bar 
                  dataKey="score" 
                  fill="#3b82f6" 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Training Time vs Performance Scatter Plot */}
      <Card>
        <CardHeader>
          <CardTitle>Training Time vs Performance</CardTitle>
          <CardDescription>
            Relationship between training time and model performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="train_time" 
                  type="number"
                  label={{ value: 'Training Time (seconds)', position: 'insideBottom', offset: -10 }}
                />
                <YAxis 
                  dataKey="score"
                  type="number"
                  label={{ value: metricName, angle: -90, position: 'insideLeft' }}
                  domain={['dataMin - 0.05', 'dataMax + 0.05']}
                />
                <Tooltip 
                  formatter={(value: number, name: string) => [
                    name === 'score' ? value.toFixed(4) : `${value.toFixed(2)}s`,
                    name === 'score' ? metricName : 'Training Time'
                  ]}
                  labelFormatter={(label, payload) => {
                    if (payload && payload[0]) {
                      return `Model: ${payload[0].payload.model}`;
                    }
                    return '';
                  }}
                />
                <Scatter 
                  dataKey="score" 
                  fill="#10b981"
                  r={8}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}