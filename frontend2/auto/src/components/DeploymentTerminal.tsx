import { useState, useMemo } from "react";
import { Terminal, Code2, Server, Play, Copy, Check, FileJson, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";
import { predictSingle } from "@/lib/api";
import { Button } from "@/components/ui/button";

interface DeploymentTerminalProps {
  modelPath: string;
  targetColumn: string;
  insights: any[];
  analysisData: any;
}

export function DeploymentTerminal({ modelPath, targetColumn, insights, analysisData }: DeploymentTerminalProps) {
  const [activeTab, setActiveTab] = useState<"api" | "docker">("api");
  const [copied, setCopied] = useState(false);
  const [consoleOutput, setConsoleOutput] = useState<string>("// Awaiting execution command...\n// Click Run Mock Server to ping the live backend.");
  const [running, setRunning] = useState(false);

  // Extract base name for display
  const modelFile = modelPath ? modelPath.split('/').pop() || "model.joblib" : "model.joblib";

  const { pythonFields, jsonPayload, mockFeatures } = useMemo(() => {
    let pythonStr = "";
    let jsonStr = "";
    const features: Record<string, number> = {};

    if (analysisData?.summary_statistics) {
       const stats = analysisData.summary_statistics;
       const allKeys = Object.keys(stats).filter(k => k !== targetColumn && k !== "count");
       
       // Generate ALL python model fields so the copied script works perfectly
       allKeys.forEach((key) => {
         pythonStr += `    ${key}: float\n`;
       });

       // Generate truncated JSON mapping for the UI curl preview
       const previewKeys = allKeys.slice(0, 5);
       previewKeys.forEach((key, idx) => {
         const isLast = idx === previewKeys.length - 1;
         const median = stats[key]?.['50%'] ?? 0;
         jsonStr += `    "${key}": ${median}${isLast ? '' : ','}\n`;
         features[key] = Number(median);
       });
       
       if (allKeys.length > 5) {
          jsonStr += `    // ... remaining ${allKeys.length - 5} features\n`;
       }
    } else {
       pythonStr = "    feature_1: float\n";
       jsonStr = `    "feature_1": 0.0\n`;
       features["feature_1"] = 0.0;
    }

    return { pythonFields: pythonStr, jsonPayload: jsonStr, mockFeatures: features };
  }, [targetColumn, analysisData]);

  const fastapiCode = `from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AutoPilot Model Inference")

# Load compiled heuristic matrix
model = joblib.load("${modelFile}")

class InferenceRequest(BaseModel):
${pythonFields}
@app.post("/predict")
def predict(request: InferenceRequest):
    data = pd.DataFrame([request.model_dump()])
    prediction = model.predict(data)[0]
    
    # Cast output for JSON serialization
    try:
        prob = model.predict_proba(data)[0].tolist()
        return {"${targetColumn}": int(prediction), "probabilities": prob}
    except:
        return {"${targetColumn}": int(prediction)}
`;

  const dockerCode = `FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose standard inference port
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
`;

  const curlCommand = `curl -X POST "http://production-endpoint/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
${jsonPayload}  }'`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(activeTab === "api" ? fastapiCode : dockerCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const executeMock = async () => {
    setRunning(true);
    setConsoleOutput(`> Executing mock POST /predict...\n> Sending payload from port 8000...\n`);
    
    try {
      // Create a payload padded with all features just like WhatIfSimulator
      const fullFeatures: Record<string, number> = {};
      if (analysisData?.summary_statistics) {
        Object.keys(analysisData.summary_statistics).forEach(feat => {
          if (feat !== targetColumn && feat !== "count") {
             fullFeatures[feat] = Number(analysisData.summary_statistics[feat]?.['50%'] ?? 0);
          }
        });
      }
      
      const start = performance.now();
      const res = await predictSingle(modelPath, fullFeatures);
      const latency = (performance.now() - start).toFixed(1);

      setConsoleOutput(prev => prev + `> HTTP 200 OK (${latency}ms)\n\n` + JSON.stringify(res, null, 2));
    } catch (err: any) {
      setConsoleOutput(prev => prev + `> HTTP 500 ERROR\n\n${err?.message || "Failed to hit inference engine."}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-2 gap-0 rounded-[2.5rem] border border-white/10 overflow-hidden shadow-2xl bg-[#0a0a0a]">
      {/* LEFT PANE - CODE GENERATION */}
      <div className="flex flex-col border-r border-white/10">
        <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-white/[0.02]">
          <div className="flex space-x-1">
             <button 
                onClick={() => setActiveTab('api')}
                className={`px-4 py-2 text-xs font-black uppercase tracking-widest rounded-md transition-all ${activeTab === 'api' ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-500 hover:text-slate-300'}`}
             >
               API.py
             </button>
             <button 
                onClick={() => setActiveTab('docker')}
                className={`px-4 py-2 text-xs font-black uppercase tracking-widest rounded-md transition-all ${activeTab === 'docker' ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-500 hover:text-slate-300'}`}
             >
               Dockerfile
             </button>
          </div>
          <button onClick={copyToClipboard} className="text-slate-500 hover:text-emerald-400 transition-colors">
            {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
          </button>
        </div>
        <div className="p-6 overflow-y-auto bg-[#050505] min-h-[350px] relative font-mono text-sm leading-relaxed text-slate-300">
           <pre className="custom-scroll">
             <code>
               {activeTab === 'api' ? fastapiCode : dockerCode}
             </code>
           </pre>
        </div>
      </div>

      {/* RIGHT PANE - SANDBOX */}
      <div className="flex flex-col bg-[#0f111a]">
         <div className="h-16 border-b border-white/5 flex items-center justify-between px-6 bg-black/20">
           <div className="flex items-center gap-3">
             <Terminal className="h-5 w-5 text-purple-400" />
             <span className="text-xs font-black text-purple-400 uppercase tracking-[0.2em]">Live Sandbox</span>
           </div>
           <Button 
             size="sm" 
             onClick={executeMock} 
             disabled={running || !modelPath}
             className="bg-purple-500 hover:bg-purple-400 text-white font-bold uppercase tracking-wider text-[10px] h-8"
           >
             {running ? <span className="animate-pulse">Running...</span> : <span className="flex items-center gap-2"><Play className="w-3 h-3"/> Test Pipeline</span>}
           </Button>
         </div>
         
         <div className="p-6 border-b border-white/5 min-h-[160px] bg-[#0a0c10]">
           <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">Example cURL Request</p>
           <pre className="text-xs font-mono text-emerald-400/80 whitespace-pre-wrap">
             {curlCommand}
           </pre>
         </div>

         <div className="flex-1 p-6 overflow-y-auto font-mono text-sm relative">
           <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">Terminal Output</p>
           <div className="text-slate-300 whitespace-pre-wrap">
             {consoleOutput}
           </div>
           <div className="fixed-bottom pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-[#0f111a] to-transparent" />
         </div>
      </div>
    </div>
  );
}
