import { useState, useCallback, useRef } from "react";
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2, Database, ShieldCheck, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "@/hooks/use-toast";
import { uploadFile } from "@/lib/api";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface UploadStepProps {
  onNext: (filepath: string, filename: string) => void;
}

export function UploadStep({ onNext }: UploadStepProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const validateAndSetFile = (selectedFile: File) => {
    if (selectedFile.name.endsWith('.csv') || selectedFile.name.endsWith('.xlsx')) {
      setFile(selectedFile);
      return true;
    } else {
      toast({
        title: "Protocol Violation",
        description: "Unsupported file format. Please use CSV or XLSX.",
        variant: "destructive",
      });
      return false;
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => (prev >= 90 ? 90 : prev + 5));
    }, 100);

    try {
      const response = await uploadFile(file);
      setProgress(100);
      clearInterval(interval);
      setTimeout(() => {
        onNext(response.filepath, response.filename);
      }, 500);
    } catch (error: any) {
      clearInterval(interval);
      setProgress(0);
      toast({
        title: "Transmission Failed",
        description: error.response?.data?.detail || "Failed to upload to remote terminal.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-5 gap-12 items-start">
      <div className="lg:col-span-3 space-y-8">
        <div className="space-y-4">
          <h2 className="text-4xl font-black text-gradient">Data Ingestion</h2>
          <p className="text-slate-500 font-medium">Initialize the pipeline by providing your source dataset.</p>
        </div>

        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            "relative group cursor-pointer transition-all duration-500 rounded-3xl border-2 border-dashed min-h-[320px] flex items-center justify-center",
            dragActive 
              ? "border-emerald-500 bg-emerald-500/5 shadow-glow scale-[1.01]" 
              : "border-white/5 bg-white/[0.02] hover:border-emerald-500/30 hover:bg-white/[0.04]"
          )}
        >
          <input 
            ref={fileInputRef}
            type="file" 
            accept=".csv,.xlsx" 
            className="hidden" 
            onChange={handleFileChange} 
          />
          
          <div className="p-12 flex flex-col items-center text-center space-y-6">
            <div className={cn(
              "w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-500",
              dragActive ? "bg-emerald-500 text-black scale-110" : "bg-white/5 text-emerald-500 group-hover:scale-110"
            )}>
              {uploading ? (
                <Loader2 className="h-10 w-10 animate-spin" />
              ) : (
                <Upload className="h-10 w-10" />
              )}
            </div>
            
            <div className="space-y-2">
              <p className="text-xl font-bold text-white uppercase tracking-tight">
                {file ? file.name : dragActive ? "Release to Ingest" : "Drop Terminal"}
              </p>
              <p className="text-sm text-slate-500 font-medium max-w-xs mx-auto">
                {file ? `${(file.size / 1024).toFixed(1)} KB recognized` : "Secure drop-zone for operational CSV or XLSX datasets."}
              </p>
            </div>

            {file && !uploading && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                    <Button 
                        size="lg" 
                        onClick={(e) => { e.stopPropagation(); handleUpload(); }}
                        className="gradient-primary h-14 px-10 font-black uppercase tracking-widest shadow-glow"
                    >
                        Begin Pipeline &rarr;
                    </Button>
                </motion.div>
            )}
          </div>

          {uploading && (
            <div className="absolute inset-x-0 bottom-0 p-8 pt-0">
               <div className="space-y-3">
                  <div className="flex justify-between text-[10px] font-black uppercase tracking-widest text-emerald-500">
                    <span>Transmitting</span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-1 bg-white/5" />
               </div>
            </div>
          )}
        </div>
      </div>

      <div className="lg:col-span-2 space-y-6">
         <div className="glass-card p-10 rounded-3xl border-white/5 space-y-8">
            <div className="flex items-center gap-3 text-emerald-500">
               <ShieldCheck className="h-5 w-5" />
               <h3 className="font-black uppercase text-xs tracking-widest">Ingestion Protocol</h3>
            </div>
            
            <ul className="space-y-6">
               {[
                 { icon: FileText, label: "Format", val: "CSV / XLSX" },
                 { icon: Database, label: "Header", val: "Mandatory Row" },
                 { icon: Zap, label: "Audit", val: "Heuristic Check" }
               ].map((item, i) => (
                 <li key={i} className="flex items-center gap-5 group">
                    <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center text-slate-500 group-hover:text-emerald-500 group-hover:bg-emerald-500/10 transition-all border border-transparent group-hover:border-emerald-500/20">
                       <item.icon className="h-5 w-5" />
                    </div>
                    <div>
                       <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">{item.label}</p>
                       <p className="text-md font-bold text-white mt-0.5">{item.val}</p>
                    </div>
                 </li>
               ))}
            </ul>

            <div className="pt-8 border-t border-white/5">
                <Alert className="bg-emerald-500/5 border-emerald-500/10 text-emerald-500/80 p-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-[10px] font-bold uppercase tracking-widest leading-loose">
                    Tip: Target column detection occurs during initial ingestion audit.
                  </AlertDescription>
                </Alert>
            </div>
         </div>
      </div>
    </div>
  );
}
