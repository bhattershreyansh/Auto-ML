import { useState } from "react";
import { Sparkles, HelpCircle } from "lucide-react";
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

  const handleComparisonNext = (comparison: any) => {
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
    <div className="min-h-screen bg-gradient-subtle">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg gradient-primary flex items-center justify-center shadow-glow">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">AutoML Platform</h1>
                <p className="text-xs text-muted-foreground">
                  Automated Machine Learning Workflow
                </p>
              </div>
            </div>
            
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <HelpCircle className="h-4 w-4 mr-2" />
                  Help
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>AutoML Platform Guide</DialogTitle>
                  <DialogDescription>
                    Follow these steps to train your machine learning model
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  {steps.map((step) => (
                    <div key={step.id} className="flex gap-3">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 border-primary text-primary font-semibold">
                        {step.id}
                      </div>
                      <div>
                        <p className="font-semibold">{step.title}</p>
                        <p className="text-sm text-muted-foreground">{step.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <WizardSteps steps={steps} currentStep={currentStep} />
          
          <Alert className="mb-6 border-primary/20 bg-primary/5">
            <Sparkles className="h-4 w-4 text-primary" />
            <AlertDescription>
              <strong>Step {currentStep} of {steps.length}:</strong> {steps[currentStep - 1]?.description}
            </AlertDescription>
          </Alert>

          <div className="bg-card rounded-xl shadow-lg p-8 border border-border">
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
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-card/50 backdrop-blur-sm mt-16">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-sm text-muted-foreground">
            Built with React, Tailwind CSS, and FastAPI • AutoML Platform © 2024
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
