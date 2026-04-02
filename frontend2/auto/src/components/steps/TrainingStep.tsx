import { useState, useEffect } from "react";
import { Loader2, CheckCircle2, Zap, Settings2, Cpu, BarChart3, Database, ShieldCheck, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { toast } from "@/hooks/use-toast";
import { selectModel, trainModel } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface TrainingStepProps {
  filepath: string;
  analysisData: any;
  onNext: (metrics: any, modelPath: string) => void;
  onBack: () => void;
}

export function TrainingStep({ filepath, analysisData, onNext, onBack }: TrainingStepProps) {
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [suggestions, setSuggestions] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [testSize, setTestSize] = useState(0.2);
  const [tuneHyperparams, setTuneHyperparams] = useState<string>("false");
  const [cvFolds, setCvFolds] = useState(5);

  useEffect(() => {
    if (analysisData?.suggested_target) {
      setTargetColumn(analysisData.suggested_target);
      loadSuggestions(analysisData.suggested_target);
    } else if (analysisData?.basic_statistics?.dtypes) {
        const firstCol = Object.keys(analysisData.basic_statistics.dtypes)[0];
        setTargetColumn(firstCol);
        loadSuggestions(firstCol);
    }
  }, []);

  const loadSuggestions = async (target: string) => {
    setLoading(true);
    try {
      const data = await selectModel(filepath, target);
      setSuggestions(data);
      if (data.recommended_models?.[0]) {
        setSelectedModel(data.recommended_models[0]);
      }
    } catch (error: any) {
      toast({
        title: "Heuristic Search Failed",
        description: error.response?.data?.detail || "Could not identify optimal models.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!selectedModel || !targetColumn) {
      toast({
        title: "Configuration Incomplete",
        description: "Protocol requires model and target selection.",
        variant: "destructive",
      });
      return;
    }

    setTraining(true);
    try {
      const tuneValue = tuneHyperparams === "false" ? false : tuneHyperparams;
      const response = await trainModel(
        filepath,
        targetColumn,
        selectedModel,
        testSize,
        tuneValue,
        cvFolds,
        50
      );
      
      toast({
        title: "Synthesis Complete",
        description: "Model weights successfully generated.",
      });
      
      onNext(response.metrics, response.model_path);
    } catch (error: any) {
      toast({
        title: "Training Aborted",
        description: error.response?.data?.detail || "Kernel execution failure.",
        variant: "destructive",
      });
    } finally {
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 space-y-8">
        <div className="relative">
            <Loader2 className="h-20 w-20 animate-spin text-emerald-500/10" strokeWidth={1} />
            <div className="absolute inset-0 flex items-center justify-center">
                <ShieldCheck className="h-8 w-8 text-emerald-500 animate-pulse" />
            </div>
            <motion.div 
                className="absolute -inset-4 rounded-full border-t-2 border-emerald-500/40"
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            />
        </div>
        <div className="text-center space-y-2">
            <p className="text-emerald-500 font-black uppercase tracking-[0.4em] text-[10px]">Benchmarking Architecture</p>
            <p className="text-slate-500 font-medium text-sm">Selecting high-performance heuristics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      <div className="text-center space-y-4">
        <h2 className="text-4xl font-black text-gradient">Model Synthesis</h2>
        <p className="text-slate-500 max-w-xl mx-auto font-medium">
          Configure architectural parameters for high-precision training.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
         <div className="lg:col-span-2 space-y-8">
            <Tabs defaultValue="simple" className="w-full">
                <TabsList className="bg-white/2 border border-white/5 p-1 rounded-2xl h-14 w-fit">
                    <TabsTrigger value="simple" className="h-full px-8 rounded-xl font-black uppercase text-[10px] tracking-widest data-[state=active]:bg-emerald-500 data-[state=active]:text-black transition-all">
                        Standard Protocol
                    </TabsTrigger>
                    <TabsTrigger value="advanced" className="h-full px-8 rounded-xl font-black uppercase text-[10px] tracking-widest data-[state=active]:bg-emerald-500 data-[state=active]:text-black transition-all">
                        Advanced Matrix
                    </TabsTrigger>
                </TabsList>

                <TabsContent value="simple" className="mt-8 space-y-6">
                    <div className="glass-card p-10 border-white/5 space-y-8">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="space-y-3">
                                <Label className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Inbound Target</Label>
                                <Select value={targetColumn} onValueChange={(val) => {
                                    setTargetColumn(val);
                                    loadSuggestions(val);
                                }}>
                                    <SelectTrigger className="h-14 bg-white/5 border-white/5 rounded-2xl text-white font-bold px-6 focus:ring-emerald-500/20">
                                        <SelectValue placeholder="Identify target" />
                                    </SelectTrigger>
                                    <SelectContent className="bg-slate-900 border-white/10 text-white">
                                        {analysisData?.basic_statistics?.dtypes && Object.keys(analysisData.basic_statistics.dtypes).map((col: string) => (
                                            <SelectItem key={col} value={col} className="hover:bg-emerald-500/10 focus:bg-emerald-500/20 focus:text-emerald-500 transition-colors">
                                              {col}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="space-y-3">
                                <Label className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Synthesis Architecture</Label>
                                <Select value={selectedModel} onValueChange={setSelectedModel}>
                                    <SelectTrigger className="h-14 bg-white/5 border-white/5 rounded-2xl text-white font-bold px-6 focus:ring-emerald-500/20">
                                        <SelectValue placeholder="Select model" />
                                    </SelectTrigger>
                                    <SelectContent className="bg-slate-900 border-white/10 text-white">
                                        {suggestions?.recommended_models?.map((model: string) => (
                                            <SelectItem key={model} value={model} className="hover:bg-emerald-500/10 focus:bg-emerald-500/20 focus:text-emerald-500 transition-colors">
                                              {model}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                    </div>
                </TabsContent>

                <TabsContent value="advanced" className="mt-8 space-y-6">
                    <div className="glass-card p-10 border-white/5 space-y-8">
                        <div className="grid md:grid-cols-2 gap-8">
                            <div className="space-y-3">
                                <Label className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Holdout Ratio</Label>
                                <Input
                                    type="number"
                                    min="0.1"
                                    max="0.5"
                                    step="0.05"
                                    value={testSize}
                                    onChange={(e) => setTestSize(parseFloat(e.target.value))}
                                    className="h-14 bg-white/5 border-white/5 rounded-2xl text-white font-bold px-6 focus:ring-emerald-500/20"
                                />
                            </div>

                            <div className="space-y-3">
                                <Label className="text-[10px] font-black uppercase text-slate-500 tracking-widest">CV Segment Folds</Label>
                                <Input
                                    type="number"
                                    min="2"
                                    max="10"
                                    value={cvFolds}
                                    onChange={(e) => setCvFolds(parseInt(e.target.value))}
                                    className="h-14 bg-white/5 border-white/5 rounded-2xl text-white font-bold px-6 focus:ring-emerald-500/20"
                                />
                            </div>

                            <div className="space-y-3">
                                <Label className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Tuning Strategy</Label>
                                <Select value={tuneHyperparams} onValueChange={setTuneHyperparams}>
                                    <SelectTrigger className="h-14 bg-white/5 border-white/5 rounded-2xl text-white font-bold px-6 focus:ring-emerald-500/20">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent className="bg-slate-900 border-white/10 text-white">
                                        <SelectItem value="false">Static Weights</SelectItem>
                                        <SelectItem value="grid">Grid Exhaustion</SelectItem>
                                        <SelectItem value="optuna">Optuna (Bayesian)</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                    </div>
                </TabsContent>
            </Tabs>
         </div>

         <div className="space-y-6">
            <div className="glass-card p-8 border-white/5 space-y-6">
               <div className="flex items-center gap-3 text-emerald-500">
                  <Activity className="h-5 w-5" />
                  <h3 className="font-black uppercase text-xs tracking-widest">Synthesis Heuristics</h3>
               </div>
               
               <ul className="space-y-6">
                  {[
                    { icon: BarChart3, label: "Kernel Task", val: suggestions?.task_type || "DETECTION" },
                    { icon: Database, label: "Volume", val: suggestions?.data_size || "MEDIUM" },
                    { icon: Settings2, label: "Heuristics", val: "Optimized" }
                  ].map((item, i) => (
                    <li key={i} className="flex items-center gap-4">
                       <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-slate-500 border border-white/5">
                          <item.icon className="h-4 w-4" />
                       </div>
                       <div>
                          <p className="text-[9px] font-black uppercase text-slate-600 tracking-widest">{item.label}</p>
                          <p className="text-sm font-bold text-white uppercase">{item.val}</p>
                       </div>
                    </li>
                  ))}
               </ul>
            </div>

            <div className="p-8 rounded-3xl bg-emerald-500/5 border border-emerald-500/10">
               <p className="text-[10px] font-bold text-emerald-500/80 leading-relaxed uppercase tracking-wider">
                  Heuristic analysis recommends <span className="text-emerald-500 font-black">{selectedModel}</span> for this payload architecture.
               </p>
            </div>
         </div>
      </div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-10 rounded-[2.5rem] bg-emerald-500/[0.03] border border-emerald-500/10 flex flex-col lg:flex-row items-center gap-10"
      >
        <div className="flex-1 space-y-3">
           <div className="flex items-center gap-2 text-emerald-500 font-black text-[10px] uppercase tracking-[0.3em]">
              <Cpu className="h-4 w-4" />
              Engine Initialization
           </div>
           <h3 className="text-xl font-bold text-white">Execute Synthesis Protocol?</h3>
           <p className="text-slate-500 text-sm font-medium max-w-xl">
              Initiate kernel execution for training the <span className="text-white font-bold">{selectedModel}</span> architecture on the provided feature set.
           </p>
        </div>
        
        <Button
          size="lg"
          onClick={handleTrain}
          disabled={training || !selectedModel || !targetColumn}
          className={cn(
            "h-16 px-12 font-black uppercase tracking-widest transition-all duration-500 min-w-[240px]",
            training ? "bg-emerald-500/20 text-emerald-500 border-none animate-pulse" : "gradient-primary shadow-glow border-none"
          )}
        >
          {training ? (
            <>
              <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              Synthesizing...
            </>
          ) : (
            <>
              Initialize Training &rarr;
            </>
          )}
        </Button>
      </motion.div>

      <div className="flex justify-between items-center pt-8">
        <Button variant="outline" onClick={onBack} disabled={training} className="glass-morphism border-white/10 px-8 h-12 font-bold uppercase text-[10px] tracking-widest">
          Back
        </Button>
      </div>
    </div>
  );
}
