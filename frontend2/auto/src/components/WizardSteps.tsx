import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

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
    <nav aria-label="Progress" className="w-full">
      <ol className="flex items-center justify-between relative px-4">
        {/* Foundation Track */}
        <div className="absolute top-[18px] left-8 right-8 h-[1px] bg-white/5 z-0" />
        
        {steps.map((step, stepIdx) => {
          const isCompleted = step.id < currentStep;
          const isActive = step.id === currentStep;

          return (
            <li
              key={step.id}
              className={cn(
                "relative flex flex-col items-center z-10",
                stepIdx !== steps.length - 1 ? "flex-1" : ""
              )}
            >
              {/* Active Progress Line */}
              {stepIdx !== steps.length - 1 && (
                <div className="absolute left-1/2 top-[18px] h-[1px] w-full z-0 overflow-hidden">
                  <motion.div
                    initial={{ width: "0%" }}
                    animate={{ width: isCompleted ? "100%" : "0%" }}
                    transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                    className="h-full bg-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.3)]"
                  />
                </div>
              )}

              <div className="relative flex flex-col items-center group">
                <motion.div
                  initial={false}
                  animate={{
                    scale: isActive ? 1.1 : 1,
                    backgroundColor: isCompleted ? "hsl(var(--primary))" : isActive ? "transparent" : "rgba(15, 23, 42, 0.8)",
                    borderColor: isCompleted || isActive ? "hsl(var(--primary))" : "rgba(255,255,255,0.05)",
                  }}
                  className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-lg border transition-all duration-500",
                    isActive && "shadow-glow bg-emerald-500/10",
                    isCompleted && "shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                  )}
                >
                  {isCompleted ? (
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}>
                      <Check className="h-4 w-4 text-black stroke-[3px]" />
                    </motion.div>
                  ) : (
                    <span className={cn(
                      "text-[10px] font-black tracking-tighter",
                      isActive ? "text-emerald-500" : "text-slate-600"
                    )}>
                      0{step.id}
                    </span>
                  )}
                  
                  {isActive && (
                    <motion.div 
                      layoutId="step-indicator"
                      className="absolute -inset-1 rounded-xl bg-emerald-500/5 border border-emerald-500/20 z-[-1]"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </motion.div>

                <div className="mt-4 text-center">
                  <motion.p
                    animate={{ 
                      color: isActive ? "#34d399" : isCompleted ? "#059669" : "#475569",
                      opacity: isActive || isCompleted ? 1 : 0.6
                    }}
                    className="text-[9px] font-black uppercase tracking-[0.2em]"
                  >
                    {step.title}
                  </motion.p>
                  <p className={cn(
                    "text-[8px] font-medium mt-1 hidden lg:block transition-colors",
                    isActive ? "text-slate-400" : "text-slate-600"
                  )}>
                    {step.description}
                  </p>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
