import { Download, CheckCircle2, BarChart3, Trophy, Sparkles, BrainCircuit, ShieldCheck, Cpu, ArrowLeftRight, FileJson } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { motion, AnimatePresence } from "framer-motion";
import { SHAPChart } from "@/components/SHAPChart";
import { cn } from "@/lib/utils";

interface ResultsStepProps {
  metrics: any;
  comparisonData?: any;
  onRestart: () => void;
}

export function ResultsStep({ metrics, comparisonData, onRestart }: ResultsStepProps) {
  const handleExport = () => {
    const results = {
      training_metrics: metrics,
      comparison: comparisonData,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `autopilot-results-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const insights = metrics?.insights || [];

  return (
    <div className="space-y-12">
      <div className="text-center space-y-4">
        <motion.div 
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
          className="inline-flex items-center justify-center w-24 h-24 rounded-[2.5rem] bg-emerald-500/10 border border-emerald-500/20 mb-4 shadow-glow"
        >
          <ShieldCheck className="h-12 w-12 text-emerald-500" />
        </motion.div>
        <h2 className="text-4xl font-black text-gradient uppercase tracking-tight">
          {metrics?.meta?.model ? `${metrics.meta.model} Authorized` : "Deployment Authorized"}
        </h2>
        <p className="text-slate-500 max-w-xl mx-auto font-medium">
          The AutoML pipeline has finalized kernel execution. Your high-performance model is validated and authorized for operational deployment.
        </p>
      </div>

      <div className="grid lg:grid-cols-5 gap-8">
        {/* Performance Metrics */}
        <motion.div
           initial={{ opacity: 0, x: -20 }}
           animate={{ opacity: 1, x: 0 }}
           className="lg:col-span-3 glass-card p-10 border-white/5 space-y-10"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-emerald-500 border border-emerald-500/10">
                <BarChart3 className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Validation Matrix</h3>
                <p className="text-[10px] text-slate-500 font-medium italic">Final Performance Heuristics</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
               <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-glow" />
               <span className="text-[10px] font-black uppercase text-emerald-500 tracking-widest">
                 Validated Champion &bull; {metrics?.meta?.model || "Active"}
               </span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
            {Object.entries(metrics || {})
              .filter(([key, value]) => 
                typeof value === 'number' && 
                !['support'].includes(key) && 
                !key.includes('avg')
              )
              .map(([key, value]: [string, any]) => (
                <div key={key} className="p-6 rounded-3xl bg-white/[0.03] border border-white/5 space-y-2 group hover:border-emerald-500/30 transition-all">
                  <p className="text-[9px] font-black uppercase text-slate-500 tracking-widest group-hover:text-slate-400 transition-colors">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-3xl font-black text-white">
                    {value.toFixed(4)}
                  </p>
                </div>
              ))}
          </div>
        </motion.div>

        {/* Insight Engine (SHAP) */}
        <motion.div
           initial={{ opacity: 0, x: 20 }}
           animate={{ opacity: 1, x: 0 }}
           className="lg:col-span-2 glass-card p-10 border-white/5 space-y-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-400 border border-purple-500/10">
                <BrainCircuit className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">XAI Diagnostics</h3>
                <p className="text-[10px] text-slate-500 font-medium italic">SHAP Intensity Weights</p>
              </div>
            </div>
            <Badge variant="outline" className="border-purple-500/20 text-purple-400 font-black uppercase text-[9px] tracking-widest h-6 px-3">Live</Badge>
          </div>

          <div className="min-h-[300px] flex flex-col justify-center">
            {insights.length > 0 ? (
              <SHAPChart data={insights} />
            ) : (
              <div className="flex flex-col items-center justify-center text-center space-y-6">
                 <div className="relative">
                    <Sparkles className="w-12 h-12 text-slate-800 animate-pulse" />
                    <div className="absolute -inset-4 bg-purple-500/5 blur-2xl rounded-full" />
                 </div>
                 <p className="text-slate-600 text-[10px] font-black uppercase tracking-[0.2em] max-w-[200px]">Synchronizing Deep Intelligence Matrix...</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {comparisonData?.leaderboard && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-10 border-white/5 space-y-10"
        >
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-amber-500/10 flex items-center justify-center text-amber-500 border border-amber-500/10">
               <Trophy className="h-6 w-6" />
            </div>
            <div>
              <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Final Operational Ranking</h3>
              <p className="text-[10px] text-slate-500 font-medium italic">Experiment Alpha-Beta Sequence</p>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {comparisonData.leaderboard.slice(0, 3).map((model: any, idx: number) => (
              <div
                key={idx}
                className={cn(
                  "flex items-center gap-6 p-6 rounded-3xl border transition-all group",
                  idx === 0 
                    ? "bg-emerald-500/[0.03] border-emerald-500/20 shadow-glow" 
                    : "bg-white/[0.02] border-white/5 hover:border-white/10"
                )}
              >
                <div className={cn(
                  "w-12 h-12 rounded-2xl flex items-center justify-center font-black text-xl border",
                  idx === 0 
                    ? "bg-emerald-500 text-black border-emerald-500 shadow-glow" 
                    : "bg-white/5 text-slate-500 border-white/5"
                )}>
                  0{idx + 1}
                </div>
                <div>
                  <p className="font-black text-white uppercase text-xs tracking-wider mb-1 group-hover:text-emerald-500 transition-colors">{model.model_name}</p>
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                    Score: <span className="text-white">{(model.score * 100).toFixed(2)}%</span>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Manifest & Deploy Activity */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="p-12 rounded-[3rem] bg-emerald-500/[0.03] border border-emerald-500/10 relative overflow-hidden group"
      >
        <div className="flex flex-col lg:flex-row items-center justify-between gap-12 relative z-10">
          <div className="text-center lg:text-left space-y-3">
            <div className="flex items-center justify-center lg:justify-start gap-3">
               <Cpu className="h-5 w-5 text-emerald-500" />
               <p className="text-emerald-500 font-black uppercase text-[10px] tracking-[0.4em]">Operational Sequence Finalized</p>
            </div>
            <h3 className="text-2xl font-black text-white uppercase tracking-tight">Generate Experiment Manifest?</h3>
            <p className="text-slate-500 text-sm font-medium max-w-xl">
               Export the high-performance weight matrix and validation heuristics for production integration.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-6 w-full lg:w-auto">
            <Button
              variant="outline"
              onClick={handleExport}
              className="h-16 px-10 glass-morphism border-white/10 font-black uppercase text-[10px] tracking-widest hover:bg-white/5 min-w-[200px]"
            >
              <FileJson className="mr-3 h-4 w-4 text-emerald-500" />
              Download Manifest
            </Button>
            <Button
              onClick={onRestart}
              className="h-16 px-12 gradient-primary font-black uppercase tracking-[0.2em] shadow-glow border-none min-w-[240px]"
            >
              Restart Pipeline &rarr;
            </Button>
          </div>
        </div>
      </motion.div>

      <div className="flex items-center justify-center gap-4 text-[10px] text-slate-800 font-black uppercase tracking-[0.5em] pt-12 pb-6">
        <ArrowLeftRight className="h-3 w-3" />
        AutoPilot Neural Engine v2.0.4
        <ArrowLeftRight className="h-3 w-3" />
      </div>
    </div>
  );
}
