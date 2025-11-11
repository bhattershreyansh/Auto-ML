import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export interface Step {
  id: number;
  title: string;
  description: string;
}

interface WizardStepsProps {
  steps: Step[];
  currentStep: number;
}

export function WizardSteps({ steps, currentStep }: WizardStepsProps) {
  return (
    <nav aria-label="Progress" className="mb-8">
      <ol className="flex items-center justify-between">
        {steps.map((step, stepIdx) => (
          <li
            key={step.id}
            className={cn(
              "relative flex flex-col items-center",
              stepIdx !== steps.length - 1 ? "flex-1" : ""
            )}
          >
            {stepIdx !== steps.length - 1 && (
              <div
                className={cn(
                  "absolute left-1/2 top-5 h-0.5 w-full transition-smooth",
                  step.id < currentStep
                    ? "bg-primary"
                    : "bg-border"
                )}
                style={{ marginLeft: "50%" }}
              />
            )}
            
            <div className="relative flex flex-col items-center z-10">
              <span
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-full border-2 transition-smooth",
                  step.id < currentStep
                    ? "border-primary bg-primary text-primary-foreground shadow-glow"
                    : step.id === currentStep
                    ? "border-primary bg-card text-primary shadow-md"
                    : "border-border bg-card text-muted-foreground"
                )}
              >
                {step.id < currentStep ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <span className="font-semibold">{step.id}</span>
                )}
              </span>
              
              <div className="mt-3 text-center min-w-[120px]">
                <p
                  className={cn(
                    "text-sm font-medium transition-smooth",
                    step.id <= currentStep ? "text-foreground" : "text-muted-foreground"
                  )}
                >
                  {step.title}
                </p>
                <p className="text-xs text-muted-foreground mt-1 hidden sm:block">
                  {step.description}
                </p>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </nav>
  );
}
