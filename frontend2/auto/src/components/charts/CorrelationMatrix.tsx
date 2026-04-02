import { cn } from "@/lib/utils";

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
  const getCorrelationValue = (x: string, y: string): number => {
    const entry = data.find(d => d.x === x && d.y === y);
    return entry ? entry.value : 0;
  };

  const getColorIntensity = (value: number): string => {
    const absValue = Math.abs(value);
    if (value > 0) {
      if (absValue >= 0.8) return "bg-emerald-500 shadow-[0_0_15px_-3px_rgba(16,185,129,0.4)]";
      if (absValue >= 0.6) return "bg-emerald-500/80";
      if (absValue >= 0.4) return "bg-emerald-500/60";
      if (absValue >= 0.2) return "bg-emerald-500/30";
      return "bg-emerald-500/10";
    } else {
      if (absValue >= 0.8) return "bg-cyan-500 shadow-[0_0_15px_-3px_rgba(6,182,212,0.4)]";
      if (absValue >= 0.6) return "bg-cyan-500/80";
      if (absValue >= 0.4) return "bg-cyan-500/60";
      if (absValue >= 0.2) return "bg-cyan-500/30";
      return "bg-cyan-500/10";
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="glass-card p-6 border-white/5 relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
           <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 7v10c0 2.21 3.58 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.58 4 8 4s8-1.79 8-4M4 7c0-2.21 3.58-4 8-4s8 1.79 8 4m0 5c0 2.21-3.58 4-8 4s-8-1.79-8-4" />
           </svg>
        </div>
        
        <div className="mb-8">
          <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-emerald-500 mb-1">Statistical Audit</h4>
          <h2 className="text-xl font-black text-white uppercase tracking-tight">Feature Correlation Matrix</h2>
          <p className="text-[11px] text-slate-500 font-medium mt-1">Relationship intensities mapped via Emerald (Positive) and Cyan (Negative) sensors.</p>
        </div>

        <div className="overflow-x-auto pb-4 custom-scrollbar">
          <div className="inline-block min-w-full">
            <div className="grid gap-1" style={{ gridTemplateColumns: `140px repeat(${columns.length}, 38px)` }}>
              {/* Header row */}
              <div />
              {columns.map((col) => (
                <div key={col} className="text-[9px] font-black text-slate-500 text-center p-1 transform -rotate-45 origin-bottom-left h-24 flex items-end justify-start pl-2">
                  <span className="whitespace-nowrap uppercase tracking-widest">{col.length > 8 ? col.substring(0, 8) + '..' : col}</span>
                </div>
              ))}
              
              {/* Matrix rows */}
              {columns.map((row) => (
                <div key={row} className="contents">
                  <div className="text-[10px] font-black p-2 text-right pr-4 flex items-center justify-end text-slate-400 uppercase tracking-tighter">
                    {row.length > 15 ? row.substring(0, 15) + '..' : row}
                  </div>
                  {columns.map((col) => {
                    const value = getCorrelationValue(col, row);
                    return (
                      <div
                        key={`${row}-${col}`}
                        className={cn(
                          "w-[38px] h-[38px] flex items-center justify-center text-[9px] font-black border border-white/[0.02] rounded-sm transition-all duration-300 cursor-crosshair z-10",
                          getColorIntensity(value),
                          "hover:z-20 hover:scale-125 hover:border-white/20 hover:text-white text-white/40"
                        )}
                        title={`${row} vs ${col}: ${value.toFixed(3)}`}
                      >
                        {Math.abs(value) >= 0.2 ? value.toFixed(2) : ''}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Technical Legend */}
        <div className="mt-8 pt-6 border-t border-white/5 flex items-center justify-center space-x-8 text-[9px] font-black uppercase tracking-widest">
          <div className="flex items-center space-x-2">
            <div className="w-2.5 h-2.5 rounded-sm bg-cyan-500 shadow-glow" />
            <span className="text-slate-500">Inverse Correlation</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2.5 h-2.5 rounded-sm bg-emerald-500/10 border border-white/5" />
            <span className="text-slate-500">Equilibrium</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2.5 h-2.5 rounded-sm bg-emerald-500 shadow-glow" />
            <span className="text-slate-500">Direct Correlation</span>
          </div>
        </div>
      </div>

      {/* Target correlations */}
      {targetCorrelations && targetCorrelations.length > 0 && (
        <div className="glass-card p-6 border-white/5">
          <div className="mb-6">
            <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-cyan-400 mb-1">Target Affinity</h4>
            <h2 className="text-lg font-black text-white uppercase tracking-tight">Top Feature Dependencies</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {targetCorrelations.slice(0, 6).map((item, idx) => (
              <div key={idx} className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] transition-colors">
                <div className="flex justify-between items-center px-1">
                  <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">{item.feature}</span>
                  <span className="text-[10px] font-mono font-bold text-white tabular-nums">
                    {item.correlation > 0 ? '+' : ''}{item.correlation.toFixed(3)}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                   <div 
                      className={cn(
                        "h-full rounded-full transition-all duration-1000",
                        item.correlation > 0 ? "bg-emerald-500 shadow-glow" : "bg-cyan-500 shadow-glow"
                      )}
                      style={{ width: `${Math.abs(item.correlation) * 100}%` }}
                   />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}