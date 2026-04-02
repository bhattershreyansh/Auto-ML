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

    // Dynamically import Plotly for performance
    import('plotly.js-dist-min').then((Plotly) => {
      try {
        const plotData = JSON.parse(data);
        
        // Define Obsidian Theme overrides
        const themeLayout = {
          ...plotData.layout,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          margin: { t: 40, r: 20, l: 50, b: 50 },
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 10,
            color: 'rgba(255, 255, 255, 0.4)'
          },
          xaxis: {
            ...plotData.layout?.xaxis,
            gridcolor: 'rgba(255, 255, 255, 0.03)',
            zerolinecolor: 'rgba(255, 255, 255, 0.05)',
            linecolor: 'rgba(255, 255, 255, 0.05)',
            tickfont: { color: 'rgba(255, 255, 255, 0.3)', size: 9 },
            title: {
               ...plotData.layout?.xaxis?.title,
               font: { size: 10, color: 'rgba(255, 255, 255, 0.5)', weight: 900 }
            }
          },
          yaxis: {
            ...plotData.layout?.yaxis,
            gridcolor: 'rgba(255, 255, 255, 0.03)',
            zerolinecolor: 'rgba(255, 255, 255, 0.05)',
            linecolor: 'rgba(255, 255, 255, 0.05)',
            tickfont: { color: 'rgba(255, 255, 255, 0.3)', size: 9 },
            title: {
               ...plotData.layout?.yaxis?.title,
               font: { size: 10, color: 'rgba(255, 255, 255, 0.5)', weight: 900 }
            }
          },
          showlegend: plotData.layout?.showlegend ?? false,
          legend: {
            font: { size: 9, color: 'rgba(255, 255, 255, 0.4)' },
            bgcolor: 'rgba(0,0,0,0)'
          }
        };

        // Create the plot with theme
        Plotly.newPlot(plotRef.current!, plotData.data, themeLayout, {
          responsive: true,
          displayModeBar: false,
          useResizeHandler: true
        });
      } catch (error) {
        console.error('Error rendering Plotly chart:', error);
      }
    }).catch((error) => console.error('Error loading Plotly:', error));

    return () => {
      if (plotRef.current) {
        plotRef.current.innerHTML = '';
      }
    };
  }, [data]);

  return (
    <div className={`w-full h-full flex flex-col ${className}`}>
      {title && (
        <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-2 text-center">
          {title}
        </h4>
      )}
      <div 
        ref={plotRef} 
        className="w-full h-full min-h-[300px] flex-1"
      />
    </div>
  );
}