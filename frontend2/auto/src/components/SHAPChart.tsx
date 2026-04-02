import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

interface SHAPInsight {
  feature: string;
  importance: number;
}

interface SHAPChartProps {
  data: SHAPInsight[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-4 border-emerald-500/10 shadow-glow backdrop-blur-xl">
        <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-2">Feature Weight</p>
        <p className="text-sm font-black text-white mb-1 uppercase tracking-wider">{label}</p>
        <div className="flex items-center gap-2 mt-2">
           <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-glow" />
           <p className="text-[11px] text-emerald-500 font-black tracking-widest">
             Intensity <span className="text-white ml-2">{(payload[0].value * 100).toFixed(4)}%</span>
           </p>
        </div>
      </div>
    );
  }
  return null;
};

export const SHAPChart: React.FC<SHAPChartProps> = ({ data }) => {
  // Sort data and take top 10 for maximum visual density
  const chartData = [...data]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);

  // Dynamic height based on data density to avoid "too much free space"
  const chartHeight = Math.max(chartData.length * 45 + 60, 200);

  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      style={{ height: chartHeight }}
      className="w-full mt-4 relative group"
    >
      <div className="absolute inset-0 bg-emerald-500/[0.01] rounded-[1.5rem] border border-emerald-500/5 -z-10 group-hover:bg-emerald-500/[0.02] transition-all" />
      
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 40, right: 40, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="barGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="rgba(16, 185, 129, 0.1)" />
              <stop offset="100%" stopColor="rgba(16, 185, 129, 0.8)" />
            </linearGradient>
            <filter id="glow">
               <feGaussianBlur stdDeviation="2" result="blur" />
               <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>
          <CartesianGrid 
            strokeDasharray="4 4" 
            horizontal={true} 
            vertical={false} 
            stroke="rgba(255,255,255,0.03)" 
          />
          <XAxis 
            type="number" 
            hide 
          />
          <YAxis 
            dataKey="feature" 
            type="category" 
            width={130}
            axisLine={false}
            tickLine={false}
            tick={{ 
              fill: 'rgba(255,255,255,0.4)', 
              fontSize: 9, 
              fontWeight: 900, 
              textAnchor: 'end', 
              dx: -5 
            }}
            className="uppercase tracking-widest font-black"
          />
          <Tooltip 
            content={<CustomTooltip />} 
            cursor={{ fill: 'rgba(16, 185, 129, 0.05)', radius: 8 }} 
            animationDuration={200}
          />
          <Bar 
            dataKey="importance" 
            radius={[0, 10, 10, 0]}
            barSize={20}
            isAnimationActive={true}
            animationDuration={1500}
            animationEasing="ease-out"
          >
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={`url(#barGradient)`}
                className="hover:stroke-emerald-400/50 hover:stroke-2 transition-all cursor-crosshair"
                style={{ filter: index === 0 ? 'url(#glow)' : 'none' }}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      {/* Precision Legend Component */}
      <div className="absolute top-4 left-6 flex items-center gap-4">
         <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500/40" />
            <span className="text-[8px] font-black uppercase text-slate-600 tracking-widest">Weighting Intensity</span>
         </div>
         <div className="flex items-center gap-1.5">
            <div className="w-4 h-0.5 bg-emerald-500/20" />
            <span className="text-[8px] font-black uppercase text-slate-600 tracking-widest">Confidence Interval</span>
         </div>
      </div>
    </motion.div>
  );
};
