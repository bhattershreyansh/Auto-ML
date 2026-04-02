import { useState } from "react";
import { Sparkles, HelpCircle, LayoutDashboard, History } from "lucide-react";
import { WizardSteps, Step } from "@/components/WizardSteps";
import { UploadStep } from "@/components/steps/UploadStep";
import { CleaningStep } from "@/components/steps/CleaningStep";
import { EDAStep } from "@/components/steps/EDAStep";
import { TrainingStep } from "@/components/steps/TrainingStep";
import { ComparisonStep } from "@/components/steps/ComparisonStep";
import { ResultsStep } from "@/components/steps/ResultsStep";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { motion, AnimatePresence } from "framer-motion";
import { Header } from "@/components/Header";

const steps: Step[] = [
  { id: 1, title: "Upload", description: "Upload dataset" },
  { id: 2, title: "Clean", description: "Data cleaning" },
  { id: 3, title: "Analyze", description: "EDA & insights" },
  { id: 4, title: "Train", description: "Model training" },
  { id: 5, title: "Compare", description: "Model comparison" },
  { id: 6, title: "Results", description: "Final results" },
];

const Index = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [filepath, setFilepath] = useState("");
  const [filename, setFilename] = useState("");
  const [cleanedFilepath, setCleanedFilepath] = useState("");
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [modelPath, setModelPath] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [comparisonData, setComparisonData] = useState<any>(null);

  const handleUploadNext = (path: string, name: string) => {
    setFilepath(path);
    setFilename(name);
    setCurrentStep(2);
  };

  const handleCleaningNext = (cleaned: string, analysis: any) => {
    setCleanedFilepath(cleaned);
    setAnalysisData(analysis);
    setCurrentStep(3);
  };

  const handleEDANext = () => {
    setCurrentStep(4);
  };

  const handleTrainingNext = (trainMetrics: any, path: string) => {
    setMetrics(trainMetrics);
    setModelPath(path);
    if (analysisData?.suggested_target) {
      setTargetColumn(analysisData.suggested_target);
    }
    setCurrentStep(5);
  };

  const handleComparisonNext = (comparison: any, electedModelName?: string) => {
    if (comparison.newMetrics) {
      setMetrics(comparison.newMetrics);
    }
    if (comparison.newModelPath) {
      setModelPath(comparison.newModelPath);
    }
    setComparisonData(comparison);
    setCurrentStep(6);
  };

  const handleRestart = () => {
    setCurrentStep(1);
    setFilepath("");
    setFilename("");
    setCleanedFilepath("");
    setAnalysisData(null);
    setMetrics(null);
    setModelPath("");
    setTargetColumn("");
    setComparisonData(null);
  };

  return (
    <div className="min-h-screen text-white relative bg-[#020617]">
      <Header />
      
      {/* Background Glows (Emerald/Carbon vibe) */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[600px] h-[600px] bg-emerald-500/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-[120px]" />
      </div>

      {/* Main Content */}
      <main className="container relative z-10 mx-auto px-6 pt-32 pb-20">
        <div className="max-w-6xl mx-auto space-y-12">
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col md:flex-row md:items-end justify-between gap-8"
          >
            <div>
              <div className="flex items-center gap-2 text-emerald-500 font-black mb-3 tracking-[0.2em] text-[10px] uppercase">
                <LayoutDashboard className="w-4 h-4" />
                Operational Cockpit
              </div>
              <h2 className="text-5xl font-black text-gradient leading-tight">Precision AutoML</h2>
              <p className="text-slate-500 mt-3 font-medium text-lg max-w-xl">
                Execute end-to-end machine learning pipelines with high-performance heuristics.
              </p>
            </div>

            <div className="flex gap-4">
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" className="glass-morphism border-white/5 h-12 px-6 font-bold uppercase text-[10px] tracking-widest hover:bg-white/5">
                    <HelpCircle className="h-4 w-4 mr-2 text-emerald-500" />
                    Standard Protocol
                  </Button>
                </DialogTrigger>
                <DialogContent className="glass-card border-white/5 max-w-2xl text-white">
                  <DialogHeader>
                    <DialogTitle className="text-2xl font-black uppercase tracking-tight">Pipeline Matrix</DialogTitle>
                    <DialogDescription className="text-slate-500 font-medium">
                      Operational stages of the AutoPilot ML environment.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
                    {steps.map((step) => (
                      <div key={step.id} className="p-5 rounded-2xl bg-white/5 border border-white/5 flex gap-4 hover:border-emerald-500/50 transition-colors group">
                        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-500 font-black text-lg group-hover:scale-110 transition-transform">
                          {step.id}
                        </div>
                        <div>
                          <p className="font-bold text-white uppercase text-[12px] tracking-wider">{step.title}</p>
                          <p className="text-xs text-slate-500 font-medium mt-1 leading-relaxed">{step.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </DialogContent>
              </Dialog>

              <Button variant="outline" className="glass-morphism border-white/5 h-12 px-6 font-bold uppercase text-[10px] tracking-widest hover:bg-white/5" onClick={() => window.location.reload()}>
                <History className="h-4 w-4 mr-2 text-cyan-400" />
                Session History
              </Button>
            </div>
          </motion.div>

          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
            >
              <WizardSteps steps={steps} currentStep={currentStep} />
            </motion.div>
            
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ type: "spring", stiffness: 400, damping: 40 }}
              className="glass-card rounded-[2.5rem] p-12 relative overflow-visible shadow-glow"
            >
              {/* Strategic Accent Line */}
              <div className="absolute top-0 left-12 right-12 h-[2px] bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent" />
              
              <div className="min-h-[400px]">
                {currentStep === 1 && <UploadStep onNext={handleUploadNext} />}
                {currentStep === 2 && (
                  <CleaningStep
                    filepath={filepath}
                    onNext={handleCleaningNext}
                    onBack={() => setCurrentStep(1)}
                  />
                )}
                {currentStep === 3 && (
                  <EDAStep
                    analysisData={analysisData}
                    onNext={handleEDANext}
                    onBack={() => setCurrentStep(2)}
                  />
                )}
                {currentStep === 4 && (
                  <TrainingStep
                    filepath={cleanedFilepath || filepath}
                    analysisData={analysisData}
                    onNext={handleTrainingNext}
                    onBack={() => setCurrentStep(3)}
                  />
                )}
                {currentStep === 5 && (
                  <ComparisonStep
                    filepath={cleanedFilepath || filepath}
                    targetColumn={targetColumn}
                    currentMetrics={metrics}
                    onNext={handleComparisonNext}
                    onBack={() => setCurrentStep(4)}
                  />
                )}
                {currentStep === 6 && (
                  <ResultsStep
                    metrics={metrics}
                    comparisonData={comparisonData}
                    onRestart={handleRestart}
                  />
                )}
              </div>
            </motion.div>
          </div>
        </div>
      </main>

      <footer className="py-16 border-t border-white/5 bg-[#020617] relative z-10">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg- emerald-500/20 flex items-center justify-center font-black text-emerald-500">AP</div>
            <p className="text-[10px] text-slate-600 font-black uppercase tracking-[0.3em]">
              AutoPilot ML Command Center
            </p>
          </div>
          <p className="text-[10px] text-slate-700 font-black uppercase tracking-widest">
            Licensed Terminal &bull; Secure MLOps Heuristics &bull; 2024
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
