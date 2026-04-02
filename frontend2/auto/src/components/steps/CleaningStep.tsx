import { useState, useEffect } from "react";
import { Loader2, Sparkles, CheckCircle2, ShieldAlert, Cpu, Database, Activity, ArrowRight, RefreshCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "@/hooks/use-toast";
import { analyzeData, cleanData } from "@/lib/api";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface CleaningStepProps {
  filepath: string;
  onNext: (cleanedFilepath: string, analysisData: any) => void;
  onBack: () => void;
}

export function CleaningStep({ filepath, onNext, onBack }: CleaningStepProps) {
  const [loading, setLoading] = useState(true);
  const [cleaning, setCleaning] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [cleaned, setCleaned] = useState(false);

  useEffect(() => {
    loadAnalysis();
  }, [filepath]);

  const loadAnalysis = async () => {
    setLoading(true);
    try {
      const data = await analyzeData(filepath);
      setAnalysisData(data);
    } catch (error: any) {
      toast({
        title: "Protocol Fault",
        description: error.response?.data?.detail || "Initial audit failed.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleClean = async () => {
    setCleaning(true);
    try {
      const response = await cleanData(filepath);
      setCleaned(true);
      toast({
        title: "Sanitization Complete",
        description: "Dataset structural integrity verified.",
      });
      
      setTimeout(() => {
        onNext(response.cleaned_filepath, analysisData);
      }, 800);
    } catch (error: any) {
      toast({
        title: "Cleaning Aborted",
        description: error.response?.data?.detail || "Heuristic conflict detected.",
        variant: "destructive",
      });
    } finally {
      setCleaning(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 space-y-8">
        <div className="relative">
            <Loader2 className="h-20 w-20 animate-spin text-emerald-500/20" strokeWidth={1} />
            <div className="absolute inset-0 flex items-center justify-center">
                <Cpu className="h-8 w-8 text-emerald-500 animate-pulse" />
            </div>
            <motion.div 
                className="absolute -inset-4 rounded-full border-t-2 border-emerald-500/40"
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            />
        </div>
        <div className="text-center space-y-2">
            <p className="text-emerald-500 font-black uppercase tracking-[0.4em] text-[10px]">Executing Heuristic Audit</p>
            <p className="text-slate-500 font-medium text-sm">Scanning dataset structural patterns...</p>
        </div>
      </div>
    );
  }

  const nullEntries = analysisData?.basic_statistics?.nulls 
    ? Object.entries(analysisData.basic_statistics.nulls).filter(([_, count]: [string, any]) => count > 0)
    : [];

  return (
    <div className="space-y-10">
      <div className="text-center space-y-4">
        <h2 className="text-4xl font-black text-gradient">Sanitization Chamber</h2>
        <p className="text-slate-500 max-w-xl mx-auto font-medium">
          Identify and resolve structural anomalies before pipeline execution.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <motion.div 
           initial={{ opacity: 0, x: -20 }}
           animate={{ opacity: 1, x: 0 }}
           className="glass-card p-8 border-white/5 space-y-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-500">
               <Database className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-black uppercase text-xs tracking-widest text-white">Structural Overview</h3>
              <p className="text-[10px] text-slate-500 font-medium">Verified dataset dimensions</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
             <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/5">
                <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest mb-1">Payload Rows</p>
                <p className="text-2xl font-black text-white">{analysisData?.basic_statistics?.shape?.[0]?.toLocaleString() || 0}</p>
             </div>
             <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/5">
                <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest mb-1">Feature Count</p>
                <p className="text-2xl font-black text-white">{analysisData?.basic_statistics?.shape?.[1] || 0}</p>
             </div>
          </div>

          <div className="pt-4 border-t border-white/5 flex items-center justify-between">
             <span className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Selected Target</span>
             <Badge className="bg-emerald-500/10 text-emerald-500 border-none font-black text-[10px] py-1 px-3">
                {analysisData?.suggested_target || "UNDETECTED"}
             </Badge>
          </div>
        </motion.div>

        <motion.div 
           initial={{ opacity: 0, x: 20 }}
           animate={{ opacity: 1, x: 0 }}
           className="glass-card p-8 border-white/5 space-y-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-500">
               <ShieldAlert className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-black uppercase text-xs tracking-widest text-white">Anomalous Entries</h3>
              <p className="text-[10px] text-slate-500 font-medium">Missing value distribution</p>
            </div>
          </div>

          <div className="space-y-3 min-h-[140px] max-h-[140px] overflow-y-auto custom-scrollbar pr-2">
            {nullEntries.length > 0 ? (
               nullEntries.map(([col, count]: [string, any]) => (
                 <div key={col} className="flex justify-between items-center py-2 border-b border-white/5 text-sm font-medium">
                    <span className="text-slate-400">{col}</span>
                    <span className="text-amber-500 font-black">{count}</span>
                 </div>
               ))
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center space-y-2 opacity-50">
                 <CheckCircle2 className="h-8 w-8 text-emerald-500" />
                 <p className="text-[10px] font-black uppercase tracking-widest text-emerald-500">Integrity Nominal</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-10 rounded-[2.5rem] bg-emerald-500/[0.03] border border-emerald-500/10 flex flex-col lg:flex-row items-center gap-10"
      >
        <div className="flex-1 space-y-3">
           <div className="flex items-center gap-2 text-emerald-500 font-black text-[10px] uppercase tracking-[0.3em]">
              <RefreshCcw className="h-4 w-4" />
              Heuristic Sanitization
           </div>
           <h3 className="text-xl font-bold text-white">Execute Structural Cleaning?</h3>
           <p className="text-slate-500 text-sm font-medium max-w-xl">
              Automated resolution of missing values via mean/mode imputation and removal of redundant structural duplications.
           </p>
        </div>
        
        <Button
          size="lg"
          onClick={handleClean}
          disabled={cleaning || cleaned}
          className={cn(
            "h-16 px-12 font-black uppercase tracking-widest transition-all duration-500 min-w-[220px]",
            cleaned ? "bg-emerald-500/20 text-emerald-500 border-none scale-105" : "gradient-primary shadow-glow border-none"
          )}
        >
          {cleaning ? (
            <>
              <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              Sanitizing...
            </>
          ) : cleaned ? (
            <>
              <CheckCircle2 className="mr-3 h-5 w-5" />
              Verified
            </>
          ) : (
            <>
              Initialize Cleanse
            </>
          )}
        </Button>
      </motion.div>

      <div className="flex justify-between items-center pt-8">
        <Button variant="outline" onClick={onBack} disabled={cleaning} className="glass-morphism border-white/10 px-8 h-12 font-bold uppercase text-[10px] tracking-widest">
          Back
        </Button>
        <div className="flex items-center gap-4">
           {cleaned && (
             <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-2 text-emerald-500 font-bold text-[10px] uppercase tracking-widest mr-4">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                Audit Ready
             </motion.div>
           )}
           <Button
             size="lg"
             onClick={() => cleaned && onNext(filepath, analysisData)}
             disabled={!cleaned}
             className="gradient-primary px-12 h-14 font-black uppercase tracking-widest shadow-glow border-none"
           >
             Continue Audit &rarr;
           </Button>
        </div>
      </div>
    </div>
  );
}
