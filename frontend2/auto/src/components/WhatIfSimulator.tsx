import { useState, useEffect } from "react";
import { SlidersHorizontal, Activity } from "lucide-react";
import { motion } from "framer-motion";
import { predictSingle } from "@/lib/api";

interface WhatIfSimulatorProps {
  modelPath: string;
  insights: any[];
  analysisData: any;
}

export function WhatIfSimulator({ modelPath, insights, analysisData }: WhatIfSimulatorProps) {
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  // Extract top 5 features
  const topFeatures = insights?.slice(0, 5).map(i => i.feature) || [];

  useEffect(() => {
    // Initialize feature defaults based on median from summary_statistics 
    if (analysisData?.summary_statistics) {
      const initial: Record<string, number> = {};
      const stats = analysisData.summary_statistics;
      
      // We must send ALL features to the model, not just the top 5
      Object.keys(stats).forEach(feat => {
        if (feat !== analysisData?.suggested_target && feat !== "count") {
          const median = stats[feat]?.['50%'] ?? stats[feat]?.['mean'] ?? 0;
          initial[feat] = Number(median);
        }
      });
      setFeatures(initial);
      runPrediction(initial);
    }
  }, [insights, analysisData]);

  const runPrediction = async (currentFeatures: Record<string, number>) => {
    if (!modelPath) return;
    setLoading(true);
    try {
      const res = await predictSingle(modelPath, currentFeatures);
      setPrediction(res.prediction);
      
      // Calculate max probability if available
      if (res.probabilities && res.probabilities.length > 0) {
        let maxProb = Math.max(...res.probabilities);
        setProbability(maxProb);
      } else {
        setProbability(null);
      }
    } catch (err) {
      console.error("Prediction Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSliderChange = (feat: string, value: number) => {
    const updated = { ...features, [feat]: value };
    setFeatures(updated);
    runPrediction(updated);
  };

  const getMinMax = (feat: string) => {
    const stats = analysisData?.summary_statistics?.[feat];
    if (stats) {
      return { 
        min: Number(stats['min']) || 0, 
        max: Number(stats['max']) || 100 
      };
    }
    return { min: 0, max: 100 };
  };

  if (!topFeatures.length) return null;

  return (
    <div className="glass-card p-10 border-white/5 space-y-8 h-full">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 flex items-center justify-center text-cyan-400 border border-cyan-500/10">
            <SlidersHorizontal className="h-6 w-6" />
          </div>
          <div>
            <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Live Telemetry</h3>
            <p className="text-[10px] text-slate-500 font-medium italic">What-If Inference Engine</p>
          </div>
        </div>
        
        <div className="relative group">
          <div className={`w-3 h-3 rounded-full ${loading ? 'bg-cyan-400/50' : 'bg-cyan-400'} shadow-[0_0_15px_rgba(34,211,238,0.5)] transition-colors`} />
        </div>
      </div>

      <div className="space-y-6">
        {topFeatures.map(feat => {
          const { min, max } = getMinMax(feat);
          // Format based on magnitude (integers if large spread, floats if small)
          const step = (max - min) > 10 ? 1 : 0.1;

          return (
            <div key={feat} className="space-y-3 relative group">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-widest">{feat}</span>
                <span className="text-xs font-black text-cyan-400">
                  {features[feat]?.toFixed(step === 1 ? 0 : 2)}
                </span>
              </div>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={features[feat] || 0}
                onChange={(e) => handleSliderChange(feat, Number(e.target.value))}
                className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
            </div>
          );
        })}
      </div>

      <div className="pt-6 border-t border-white/5 flex flex-col items-center">
        <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500 mb-4">Live Prediction Output</h4>
        <div className="relative">
          <motion.div 
            key={String(prediction)}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={`w-32 h-32 rounded-full border-4 ${prediction === 1 ? "border-emerald-500" : "border-cyan-500"} flex flex-col items-center justify-center shadow-glow`}
          >
            <span className="text-4xl font-black text-white">
              {prediction}
            </span>
            {probability !== null && (
              <span className="text-[10px] text-slate-400 font-black tracking-widest">
                {(probability * 100).toFixed(1)}% CONF
              </span>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
