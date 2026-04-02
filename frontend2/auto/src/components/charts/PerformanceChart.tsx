import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';
import { motion } from 'framer-motion';

interface ChartData {
  model: string;
  score: number;
  train_time: number;
}

interface PerformanceChartProps {
  data: ChartData[];
  taskType: string;
}

const CustomTooltip = ({ active, payload, label, metricName }: any) => {
  if (active && payload && payload.length) {
    const modelName = payload[0].payload.model;
    return (
      <div className="glass-card p-4 border-emerald-500/10 shadow-glow backdrop-blur-xl">
        <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-2">Model Metric</p>
        <p className="text-sm font-black text-white mb-1 uppercase tracking-wider">{modelName}</p>
        <div className="flex flex-col gap-1.5 mt-2">
           <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-glow" />
              <p className="text-[11px] text-emerald-500 font-black tracking-widest">
                {metricName}: <span className="text-white ml-1">{payload[0].value.toFixed(4)}</span>
              </p>
           </div>
           {payload[0].payload.train_time && (
             <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500 shadow-glow" />
                <p className="text-[11px] text-cyan-500 font-black tracking-widest">
                  TIME: <span className="text-white ml-1">{payload[0].payload.train_time.toFixed(3)}s</span>
                </p>
             </div>
           )}
        </div>
      </div>
    );
  }
  return null;
};

export function PerformanceChart({ data, taskType }: PerformanceChartProps) {
  const metricName = taskType === 'classification' ? 'Accuracy' : 'R² Score';

  return (
    <div className="space-y-8">
      {/* Performance Bar Chart */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 border-white/5"
      >
        <div className="mb-8">
          <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-emerald-500 mb-1">Comparative Analysis</h4>
          <h2 className="text-xl font-black text-white uppercase tracking-tight">Model Performance Leaderboard</h2>
          <p className="text-[11px] text-slate-500 font-medium mt-1">Cross-model evaluation based on {metricName} benchmarks.</p>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
              <defs>
                <linearGradient id="perfGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(161 72% 48%)" stopOpacity={0.8} />
                  <stop offset="100%" stopColor="hsl(161 72% 48%)" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="4 4" horizontal={true} vertical={false} stroke="rgba(255,255,255,0.03)" />
              <XAxis 
                dataKey="model" 
                angle={-45}
                textAnchor="end"
                height={60}
                interval={0}
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 9, fontWeight: 900 }}
                className="uppercase tracking-tighter"
              />
              <YAxis 
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 9, fontWeight: 900 }}
                domain={['dataMin - 0.05', 'dataMax + 0.05']}
              />
              <Tooltip content={<CustomTooltip metricName={metricName} />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar 
                dataKey="score" 
                radius={[4, 4, 0, 0]}
                animationDuration={1500}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill="url(#perfGradient)" />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Training Time vs Performance Scatter Plot */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-6 border-white/5"
      >
        <div className="mb-8">
          <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-cyan-400 mb-1">Resource Diagnostics</h4>
          <h2 className="text-xl font-black text-white uppercase tracking-tight">Computational Efficiency</h2>
          <p className="text-[11px] text-slate-500 font-medium mt-1">Relationship between training latency and inference precision.</p>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.03)" />
              <XAxis 
                dataKey="train_time" 
                type="number"
                name="Time"
                unit="s"
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 9, fontWeight: 900 }}
                label={{ value: 'LATENCY (SECONDS)', position: 'insideBottom', offset: -10, fill: 'rgba(255,255,255,0.2)', fontSize: 8, fontWeight: 900, tracking: '0.2em' }}
              />
              <YAxis 
                dataKey="score"
                type="number"
                name="Score"
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 9, fontWeight: 900 }}
                domain={['dataMin - 0.05', 'dataMax + 0.05']}
                label={{ value: metricName.toUpperCase(), angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.2)', fontSize: 8, fontWeight: 900, tracking: '0.2em' }}
              />
              <Tooltip content={<CustomTooltip metricName={metricName} />} cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.1)' }} />
              <Scatter 
                name="Models" 
                data={data} 
                fill="hsl(189 90% 50%)"
                line={{ stroke: 'rgba(16, 185, 129, 0.1)', strokeWidth: 1 }}
                shape="circle"
              >
                {data.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.score === Math.max(...data.map(d => d.score)) ? "hsl(161 72% 48%)" : "hsl(189 90% 50%)"}
                    style={{ filter: 'drop-shadow(0 0 8px currentColor)' }}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </div>
  );
}