import { useState } from "react";
import { Trophy, Clock, Zap, Loader2, BarChart3, Activity, Target, Cpu, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/hooks/use-toast";
import { compareModels, trainModel } from "@/lib/api";
import { CorrelationMatrix } from "@/components/charts/CorrelationMatrix";
import { PerformanceChart } from "@/components/charts/PerformanceChart";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface ComparisonStepProps {
  filepath: string;
  targetColumn: string;
  currentMetrics: any;
  onNext: (comparisonData: any, electedModelName?: string) => void;
  onBack: () => void;
}

export function ComparisonStep({ filepath, targetColumn, currentMetrics, onNext, onBack }: ComparisonStepProps) {
  const [comparing, setComparing] = useState(false);
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [electedModelName, setElectedModelName] = useState<string>(currentMetrics?.meta?.model || "");
  const [activating, setActivating] = useState(false);

  const handleFinalize = async () => {
    if (electedModelName && electedModelName !== currentMetrics?.meta?.model) {
      setActivating(true);
      toast({
        title: "Activating Champion",
        description: `Synthesizing artifacts and SHAP insights for ${electedModelName}...`,
      });
      try {
        const response = await trainModel(filepath, targetColumn, electedModelName, 0.2, false, 5);
        onNext({ ...comparisonData, newMetrics: response.metrics, newModelPath: response.model_filename }, electedModelName);
      } catch (error: any) {
        toast({
          title: "Activation Aborted",
          description: error.response?.data?.detail || "Kernel failure during champion synthesis.",
          variant: "destructive",
        });
      } finally {
        setActivating(false);
      }
    } else {
      onNext(comparisonData, electedModelName);
    }
  };

  const handleCompare = async () => {
    setComparing(true);
    try {
      const response = await compareModels(filepath, targetColumn, undefined, 0.2, false, 3);
      setComparisonData(response.comparison);
      toast({
        title: "Benchmark Complete",
        description: "Multi-model evaluation matrix generated.",
      });
    } catch (error: any) {
      toast({
        title: "Benchmark Aborted",
        description: error.response?.data?.detail || "Evaluation kernel failure.",
        variant: "destructive",
      });
    } finally {
      setComparing(false);
    }
  };

  const metricEntries = currentMetrics ? Object.entries(currentMetrics)
    .filter(([key, value]) => 
      typeof value === 'number' && 
      !['support'].includes(key) && 
      !key.includes('avg')
    )
    .slice(0, 6) : [];

  return (
    <div className="space-y-12">
      <div className="text-center space-y-4">
        <h2 className="text-4xl font-black text-gradient">Performance Matrix</h2>
        <p className="text-slate-500 max-w-xl mx-auto font-medium">
          Benchmark model heuristics against industry standards and alternative architectures.
        </p>
      </div>

      {currentMetrics && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-10 border-emerald-500/10 space-y-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-emerald-500">
                <Trophy className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Current Model Metrics</h3>
                <p className="text-[10px] text-slate-500 font-medium italic">Validated Performance Heuristics</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
               <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
               <span className="text-[10px] font-black uppercase text-emerald-500 tracking-widest">Active Model</span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
            {metricEntries.map(([key, value]: [string, any]) => (
              <div key={key} className="p-5 rounded-2xl bg-white/[0.03] border border-white/5 space-y-2 group hover:border-emerald-500/30 transition-all">
                <p className="text-[9px] font-black uppercase text-slate-500 tracking-widest group-hover:text-slate-400 transition-colors">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-2xl font-black text-white">
                  {value.toFixed(4)}
                </p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {!comparisonData ? (
        <motion.div 
           initial={{ opacity: 0, scale: 0.95 }}
           animate={{ opacity: 1, scale: 1 }}
           className="glass-card p-16 text-center space-y-10 border-dashed border-2 border-white/5 bg-transparent"
        >
          <div className="space-y-4">
             <div className="w-20 h-20 rounded-3xl bg-emerald-500/5 mx-auto flex items-center justify-center">
                <Zap className="h-10 w-10 text-emerald-500" />
             </div>
             <h3 className="text-2xl font-black text-white uppercase tracking-tight">Full Pipeline Benchmark</h3>
             <p className="text-slate-500 max-w-sm mx-auto font-medium text-sm">
                Execute a parallel evaluation of multiple architectures to identify the absolute performance peak.
             </p>
          </div>

          <Button
            size="lg"
            onClick={handleCompare}
            disabled={comparing}
            className="h-16 px-12 gradient-primary font-black uppercase tracking-[0.2em] shadow-glow border-none min-w-[280px]"
          >
            {comparing ? (
              <>
                <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                Executing Matrix...
              </>
            ) : (
              "Initialize Benchmark &rarr;"
            )}
          </Button>
        </motion.div>
      ) : (
        <Tabs defaultValue="leaderboard" className="space-y-8">
          <TabsList className="bg-white/2 border border-white/5 p-1 rounded-2xl h-14 w-fit">
            <TabsTrigger value="leaderboard" className="h-full px-8 rounded-xl font-black uppercase text-[10px] tracking-widest data-[state=active]:bg-emerald-500 data-[state=active]:text-black transition-all">
              <Trophy className="h-4 w-4 mr-2" />
              Leaderboard
            </TabsTrigger>
            <TabsTrigger value="charts" className="h-full px-8 rounded-xl font-black uppercase text-[10px] tracking-widest data-[state=active]:bg-emerald-500 data-[state=active]:text-black transition-all">
              <BarChart3 className="h-4 w-4 mr-2" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="correlation" className="h-full px-8 rounded-xl font-black uppercase text-[10px] tracking-widest data-[state=active]:bg-emerald-500 data-[state=active]:text-black transition-all">
              <Activity className="h-4 w-4 mr-2" />
              Correlation
            </TabsTrigger>
          </TabsList>

          <TabsContent value="leaderboard" className="mt-8">
            <div className="glass-card overflow-hidden border-white/10 shadow-glow">
              <Table>
                <TableHeader className="bg-white/[0.03]">
                  <TableRow className="border-white/5 hover:bg-transparent">
                    <TableHead className="text-[10px] font-black uppercase text-slate-500 tracking-widest py-6 px-8">Rank</TableHead>
                    <TableHead className="text-[10px] font-black uppercase text-slate-500 tracking-widest py-6">Architecture</TableHead>
                    <TableHead className="text-[10px] font-black uppercase text-slate-500 tracking-widest py-6">Confidence Score</TableHead>
                    <TableHead className="text-[10px] font-black uppercase text-slate-500 tracking-widest py-6 px-8">Kernel Time</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {comparisonData.leaderboard?.map((model: any, idx: number) => (
                    <TableRow key={idx} onClick={() => setElectedModelName(model.model_name)} className={cn(
                        "border-white/5 transition-all cursor-pointer",
                        model.model_name === electedModelName ? "bg-emerald-500/10 border-emerald-500/50 shadow-[inset_0_0_20px_rgba(16,185,129,0.1)]" : "hover:bg-white/[0.02]"
                    )}>
                      <TableCell className="px-8 py-6">
                        {idx === 0 ? (
                          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500 text-black font-black text-xs shadow-glow">
                            01
                          </div>
                        ) : (
                          <span className="text-slate-600 font-bold ml-2">0{idx + 1}</span>
                        )}
                      </TableCell>
                      <TableCell className="font-bold text-white uppercase text-xs tracking-wider">{model.model_name}</TableCell>
                      <TableCell>
                        <span className={cn(
                            "font-black text-sm",
                            idx === 0 ? "text-emerald-500" : "text-slate-300"
                        )}>
                          {(model.score * 100).toFixed(2)}%
                        </span>
                      </TableCell>
                      <TableCell className="px-8 py-6 italic text-slate-500 text-[11px] font-medium">
                        {model.train_time?.toFixed(3)}s
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </TabsContent>

          <TabsContent value="charts" className="mt-8">
            <div className="glass-card p-10 border-white/5 min-h-[400px] flex flex-col items-center justify-center">
                {comparisonData.chart_data?.length > 0 ? (
                <PerformanceChart 
                    data={comparisonData.chart_data} 
                    taskType={comparisonData.task_type}
                />
                ) : (
                <p className="text-slate-500 font-black uppercase text-[10px] tracking-widest">Awaiting Metric Visualization</p>
                )}
            </div>
          </TabsContent>

          <TabsContent value="correlation" className="mt-8">
            <div className="glass-card p-10 border-white/5 min-h-[400px]">
                {comparisonData.correlation_data ? (
                <CorrelationMatrix 
                    data={comparisonData.correlation_data.correlation_matrix}
                    columns={comparisonData.correlation_data.columns}
                    targetCorrelations={comparisonData.correlation_data.target_correlations}
                />
                ) : (
                <div className="h-full flex items-center justify-center">
                    <p className="text-slate-500 font-black uppercase text-[10px] tracking-widest">Correlation Engine Inactive</p>
                </div>
                )}
            </div>
          </TabsContent>
        </Tabs>
      )}

      <div className="flex justify-between items-center pt-12 border-t border-white/5">
        <Button variant="outline" onClick={onBack} disabled={comparing} className="glass-morphism border-white/10 px-10 h-14 font-black uppercase text-[10px] tracking-widest hover:bg-white/5">
          Back
        </Button>
        <Button
          size="lg"
          onClick={handleFinalize}
          disabled={!comparisonData || activating}
          className="gradient-primary px-14 h-16 font-black uppercase tracking-[0.2em] shadow-glow border-none"
        >
          {activating ? (
            <>
              <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              Synthesizing...
            </>
          ) : (
            electedModelName && electedModelName !== currentMetrics?.meta?.model ? "Activate Champion & Deploy \u2192" : "Finalize & Deploy \u2192"
          )}
        </Button>
      </div>
    </div>
  );
}
