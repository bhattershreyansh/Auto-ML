import { useState, useCallback } from "react";
import { Upload, FileText, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "@/hooks/use-toast";
import { uploadFile } from "@/lib/api";
import { cn } from "@/lib/utils";

interface UploadStepProps {
  onNext: (filepath: string, filename: string) => void;
}

export function UploadStep({ onNext }: UploadStepProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a CSV file",
          variant: "destructive",
        });
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a CSV file",
          variant: "destructive",
        });
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    try {
      const response = await uploadFile(file);
      console.log('Upload response:', response);
      
      if (!response.filepath || !response.filename) {
        throw new Error('Invalid response format from server');
      }
      
      toast({
        title: "Success!",
        description: "File uploaded successfully",
      });
      onNext(response.filepath, response.filename);
    } catch (error: any) {
      console.error('Upload error:', error);
      toast({
        title: "Upload failed",
        description: error.response?.data?.detail || error.message || "Failed to upload file",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold gradient-primary bg-clip-text text-transparent">
          Upload Your Dataset
        </h2>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Start your AutoML journey by uploading a CSV file. Our system will analyze and
          prepare your data for machine learning.
        </p>
      </div>

      <Card className="p-8 transition-smooth hover:shadow-lg">
        <div
          className={cn(
            "border-2 border-dashed rounded-lg p-12 text-center transition-smooth cursor-pointer",
            dragActive
              ? "border-primary bg-primary/5 shadow-glow"
              : "border-border hover:border-primary/50 hover:bg-muted/50"
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <input
            id="file-upload"
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleFileChange}
          />
          
          <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          
          <div className="space-y-2">
            <p className="text-lg font-medium">
              {file ? (
                <span className="flex items-center justify-center gap-2 text-primary">
                  <FileText className="h-5 w-5" />
                  {file.name}
                </span>
              ) : (
                "Drag and drop your CSV file here"
              )}
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse your files
            </p>
          </div>

          {file && (
            <p className="text-xs text-muted-foreground mt-4">
              Size: {(file.size / 1024).toFixed(2)} KB
            </p>
          )}
        </div>
      </Card>

      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          <strong>Tip:</strong> Make sure your CSV file has a header row with column names.
          The system will automatically detect data types and suggest target columns for prediction.
        </AlertDescription>
      </Alert>

      <div className="flex justify-end">
        <Button
          size="lg"
          variant="gradient"
          onClick={handleUpload}
          disabled={!file || uploading}
        >
          {uploading ? "Uploading..." : "Next: Analyze Data"}
        </Button>
      </div>
    </div>
  );
}
