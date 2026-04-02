import { BarChart3, TrendingUp, Activity, PieChart, Database, Target, BrainCircuit, Columns, ShieldCheck, Microscope, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface EDAStepProps {
  analysisData: any;
  onNext: () => void;
  onBack: () => void;
}

export function EDAStep({ analysisData, onNext, onBack }: EDAStepProps) {
  const dtypes = analysisData?.basic_statistics?.dtypes || {};
  const dtypeEntries = Object.entries(dtypes);
  
  return (
    <div className="space-y-12">
      <div className="text-center space-y-4">
        <h2 className="text-4xl font-black text-gradient uppercase tracking-tight">
           Feature Engineering Audit
        </h2>
        <p className="text-slate-500 max-w-xl mx-auto font-medium">
          Deep diagnostic analysis of feature distributions and algorithmic correlations.
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Schema Audit */}
        <motion.div 
           initial={{ opacity: 0, x: -20 }}
           animate={{ opacity: 1, x: 0 }}
           className="lg:col-span-1 glass-card p-10 border-white/5 flex flex-col"
        >
          <div className="flex items-center gap-4 mb-8">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-emerald-500 border border-emerald-500/10">
               <Columns className="h-6 w-6" />
            </div>
            <div>
              <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Schema Audit</h3>
              <p className="text-[10px] text-slate-500 font-medium italic">Detected Data Types</p>
            </div>
          </div>
          
          <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar flex-1">
            {dtypeEntries.slice(0, 50).map(([col, dtype]: [string, any]) => (
              <div key={col} className="flex justify-between items-center py-3 border-b border-white/5 last:border-0 group transition-all">
                <span className="text-xs font-bold text-slate-400 group-hover:text-white transition-colors truncate max-w-[150px] uppercase tracking-wider">{col}</span>
                <Badge variant="outline" className="text-[9px] uppercase font-black border-white/10 text-emerald-500 bg-white/5">{dtype}</Badge>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Diagnostic Grid */}
        <motion.div 
           initial={{ opacity: 0, y: 20 }}
           animate={{ opacity: 1, y: 0 }}
           className="lg:col-span-2 space-y-8"
        >
          <div className="glass-card p-10 border-white/5">
             <div className="flex items-center justify-between mb-10">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-400 border border-purple-500/10">
                        <Target className="h-6 w-6" />
                    </div>
                    <div>
                        <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Optimization Profile</h3>
                        <p className="text-[10px] text-slate-500 font-medium italic">Heuristic Signature</p>
                    </div>
                </div>
                <Badge className="bg-emerald-500 text-black font-black uppercase text-[9px] tracking-widest px-4 h-6">Active</Badge>
             </div>
             
             <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="p-6 rounded-3xl bg-white/[0.03] border border-white/5 space-y-2">
                   <p className="text-[9px] text-slate-500 font-black uppercase tracking-widest">Inbound Target</p>
                   <p className="text-2xl font-black text-white capitalize truncate">
                      {analysisData?.suggested_target || "Null"}
                   </p>
                </div>
                <div className="p-6 rounded-3xl bg-white/[0.03] border border-white/5 space-y-2">
                   <p className="text-[9px] text-slate-500 font-black uppercase tracking-widest">Detected Task</p>
                   <p className="text-2xl font-black text-emerald-500 capitalize">
                      {analysisData?.task_type || "N/A"}
                   </p>
                </div>
             </div>
          </div>

          <div className="glass-card p-10 border-white/5">
             <div className="flex items-center gap-4 mb-10">
                <div className="w-12 h-12 rounded-2xl bg-amber-500/10 flex items-center justify-center text-amber-500 border border-amber-500/10">
                   <Microscope className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="font-black uppercase text-xs tracking-[0.2em] text-white">Payload Metrics</h3>
                  <p className="text-[10px] text-slate-500 font-medium italic">Dimensionality Overview</p>
                </div>
             </div>
             <div className="grid grid-cols-2 gap-8">
                <div className="space-y-1">
                   <p className="text-4xl font-black text-white leading-none">
                      {analysisData?.basic_statistics?.shape?.[0]?.toLocaleString() || 0}
                   </p>
                   <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.3em] mt-2">Samples Detected</p>
                </div>
                <div className="space-y-1">
                   <p className="text-4xl font-black text-white leading-none">
                      {analysisData?.basic_statistics?.shape?.[1] || 0}
                   </p>
                   <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.3em] mt-2">Feature Vectors</p>
                </div>
             </div>
          </div>
        </motion.div>
      </div>

      {/* Advanced Visualizations */}
      {analysisData?.visualizations && Object.keys(analysisData.visualizations).length > 0 && (
         <motion.div 
           initial={{ opacity: 0, y: 30 }}
           animate={{ opacity: 1, y: 0 }}
           className="glass-card overflow-hidden border-white/5"
         >
            <Tabs defaultValue="distributions" className="w-full">
                <TabsList className="bg-white/2 border-b border-white/5 p-0 h-16 w-full justify-start rounded-none">
                    <TabsTrigger value="distributions" className="h-full px-10 rounded-none font-black uppercase text-[10px] tracking-widest border-b-2 border-transparent data-[state=active]:border-emerald-500 data-[state=active]:bg-white/[0.02] transition-all">
                        Linear Distributions
                    </TabsTrigger>
                    <TabsTrigger value="correlations" className="h-full px-10 rounded-none font-black uppercase text-[10px] tracking-widest border-b-2 border-transparent data-[state=active]:border-emerald-500 data-[state=active]:bg-white/[0.02] transition-all">
                        Correlation Matrix
                    </TabsTrigger>
                    <TabsTrigger value="quality" className="h-full px-10 rounded-none font-black uppercase text-[10px] tracking-widest border-b-2 border-transparent data-[state=active]:border-emerald-500 data-[state=active]:bg-white/[0.02] transition-all">
                        Data Integrity Map
                    </TabsTrigger>
                </TabsList>
                
                <div className="p-10">
                    <TabsContent value="distributions" className="mt-0">
                        <div className="grid md:grid-cols-2 gap-12">
                            {analysisData.visualizations.target_distribution && (
                                <div className="space-y-6">
                                    <div className="flex items-center gap-2 text-emerald-500/80 font-black text-[9px] uppercase tracking-widest">
                                        <Info className="h-3 w-3" />
                                        Target Skew Diagnostics
                                    </div>
                                    <PlotlyChart data={analysisData.visualizations.target_distribution} />
                                </div>
                            )}
                            {analysisData.visualizations.numeric_distributions && (
                                <div className="space-y-6">
                                    <div className="flex items-center gap-2 text-emerald-500/80 font-black text-[9px] uppercase tracking-widest">
                                        <Info className="h-3 w-3" />
                                        Numeric Variance Audit
                                    </div>
                                    <PlotlyChart data={analysisData.visualizations.numeric_distributions} />
                                </div>
                            )}
                        </div>
                    </TabsContent>

                    <TabsContent value="correlations" className="mt-0">
                        {analysisData.visualizations.correlation_heatmap && (
                            <div className="space-y-6">
                                <div className="flex items-center gap-2 text-emerald-500/80 font-black text-[9px] uppercase tracking-widest">
                                    <TrendingUp className="h-3 w-3" />
                                    Multivariate Correlation Matrix
                                </div>
                                <PlotlyChart data={analysisData.visualizations.correlation_heatmap} />
                            </div>
                        )}
                    </TabsContent>

                    <TabsContent value="quality" className="mt-0">
                        {analysisData.visualizations.missing_values && (
                            <div className="space-y-6">
                                <div className="flex items-center gap-2 text-emerald-500/80 font-black text-[9px] uppercase tracking-widest">
                                    <ShieldCheck className="h-3 w-3" />
                                    Data Nullity Heatmap
                                </div>
                                <PlotlyChart data={analysisData.visualizations.missing_values} />
                            </div>
                        )}
                    </TabsContent>
                </div>
            </Tabs>
         </motion.div>
      )}

      {/* Navigation Protocols */}
      <div className="flex justify-between items-center pt-12 border-t border-white/5">
        <Button variant="outline" onClick={onBack} className="glass-morphism border-white/10 px-10 h-14 font-black uppercase text-[10px] tracking-widest hover:bg-white/5">
          Back
        </Button>
        <Button
          size="lg"
          className="gradient-primary px-14 h-16 font-black uppercase tracking-[0.2em] shadow-glow border-none"
          onClick={onNext}
        >
          Initialize Synthesis &rarr;
        </Button>
      </div>
    </div>
  );
}
