import { useEffect, useRef } from 'react';

interface PlotlyChartProps {
  data: string; // Plotly JSON string
  title?: string;
  className?: string;
}

export function PlotlyChart({ data, title, className = "" }: PlotlyChartProps) {
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!data || !plotRef.current) return;

    // Dynamically import Plotly to avoid SSR issues
    import('plotly.js-dist-min').then((Plotly) => {
      try {
        const plotData = JSON.parse(data);
        
        // Create the plot
        Plotly.newPlot(plotRef.current!, plotData.data, plotData.layout, {
          responsive: true,
          displayModeBar: false,
        });
      } catch (error) {
        console.error('Error rendering Plotly chart:', error);
        if (plotRef.current) {
          plotRef.current.innerHTML = `
            <div class="flex items-center justify-center h-full text-muted-foreground">
              <p>Error loading chart</p>
            </div>
          `;
        }
      }
    }).catch((error) => {
      console.error('Error loading Plotly:', error);
      if (plotRef.current) {
        plotRef.current.innerHTML = `
          <div class="flex items-center justify-center h-full text-muted-foreground">
            <p>Plotly not available</p>
          </div>
        `;
      }
    });

    // Cleanup function
    return () => {
      if (plotRef.current) {
        // Clear the plot
        plotRef.current.innerHTML = '';
      }
    };
  }, [data]);

  return (
    <div className={`w-full h-full ${className}`}>
      {title && (
        <h4 className="text-sm font-medium mb-2 text-center">{title}</h4>
      )}
      <div 
        ref={plotRef} 
        className="w-full h-full min-h-[300px]"
        style={{ minHeight: '300px' }}
      />
    </div>
  );
}