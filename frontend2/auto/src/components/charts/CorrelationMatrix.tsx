import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface CorrelationData {
  x: string;
  y: string;
  value: number;
}

interface CorrelationMatrixProps {
  data: CorrelationData[];
  columns: string[];
  targetCorrelations?: Array<{ feature: string; correlation: number }>;
}

export function CorrelationMatrix({ data, columns, targetCorrelations }: CorrelationMatrixProps) {
  // Create a grid for the correlation matrix
  const getCorrelationValue = (x: string, y: string): number => {
    const entry = data.find(d => d.x === x && d.y === y);
    return entry ? entry.value : 0;
  };

  const getColorIntensity = (value: number): string => {
    const absValue = Math.abs(value);
    if (absValue >= 0.8) return value > 0 ? "bg-blue-600" : "bg-red-600";
    if (absValue >= 0.6) return value > 0 ? "bg-blue-500" : "bg-red-500";
    if (absValue >= 0.4) return value > 0 ? "bg-blue-400" : "bg-red-400";
    if (absValue >= 0.2) return value > 0 ? "bg-blue-300" : "bg-red-300";
    return "bg-gray-200";
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Feature Correlation Matrix</CardTitle>
          <CardDescription>
            Correlation between features (blue = positive, red = negative)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              <div className="grid gap-1" style={{ gridTemplateColumns: `120px repeat(${columns.length}, 40px)` }}>
                {/* Header row */}
                <div></div>
                {columns.map((col) => (
                  <div key={col} className="text-xs font-medium text-center p-1 transform -rotate-45 origin-bottom-left h-16 flex items-end justify-center">
                    <span className="whitespace-nowrap">{col.length > 8 ? col.substring(0, 8) + '...' : col}</span>
                  </div>
                ))}
                
                {/* Matrix rows */}
                {columns.map((row) => (
                  <div key={row} className="contents">
                    <div className="text-xs font-medium p-2 text-right pr-2 flex items-center justify-end">
                      {row.length > 15 ? row.substring(0, 15) + '...' : row}
                    </div>
                    {columns.map((col) => {
                      const value = getCorrelationValue(col, row);
                      return (
                        <div
                          key={`${row}-${col}`}
                          className={`w-10 h-10 flex items-center justify-center text-xs font-medium text-white ${getColorIntensity(value)} hover:scale-110 transition-transform cursor-pointer`}
                          title={`${row} vs ${col}: ${value.toFixed(3)}`}
                        >
                          {Math.abs(value) >= 0.1 ? value.toFixed(2) : ''}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Legend */}
          <div className="mt-4 flex items-center justify-center space-x-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500"></div>
              <span>Negative</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-gray-200"></div>
              <span>Neutral</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500"></div>
              <span>Positive</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Target correlations */}
      {targetCorrelations && targetCorrelations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Top Feature Correlations with Target</CardTitle>
            <CardDescription>
              Features most correlated with the target variable
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {targetCorrelations.slice(0, 5).map((item, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <span className="font-medium">{item.feature}</span>
                  <div className="flex items-center space-x-2">
                    <div 
                      className={`w-20 h-2 rounded-full ${
                        item.correlation > 0 ? 'bg-blue-500' : 'bg-red-500'
                      }`}
                      style={{ 
                        width: `${Math.abs(item.correlation) * 80}px`,
                        minWidth: '10px'
                      }}
                    ></div>
                    <span className="text-sm font-semibold min-w-[60px] text-right">
                      {item.correlation.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}