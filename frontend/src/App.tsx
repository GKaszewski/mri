// src/App.tsx

import React, { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type PredictionResult = {
  label: string;
  probability: number;
  grad_cam: string; // base64 PNG string
};

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileError, setFileError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setFileError("");
    setResult(null);

    const file = inputRef.current?.files?.[0];
    if (!file) {
      setFileError("Please select a file.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Prediction failed");
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setFileError("Failed to process the image. Please try again.");
      console.error("Error during prediction:", error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-slate-300 flex items-center justify-center">
      <Card className="w-full max-w-md shadow-2xl rounded-2xl p-6">
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Label htmlFor="file">Upload MRI Image</Label>
          <Input
            type="file"
            accept="image/png, image/jpeg"
            id="file"
            ref={inputRef}
            className="border"
            disabled={loading}
          />
          {fileError && <span className="text-red-600 text-sm">{fileError}</span>}
          <Button type="submit" disabled={loading} className="mt-2">
            {loading ? "Analyzing..." : "Analyze Image"}
          </Button>
        </form>

        {result && (
          <CardContent className="mt-6">
            <h2 className="text-lg font-bold mb-2">Prediction Result</h2>
            <div className="flex items-center gap-2 mb-2">
              <span
                className={`inline-block px-3 py-1 rounded-xl text-white ${
                  result.label === "Tumor" ? "bg-red-600" : "bg-green-600"
                }`}
              >
                {result.label}
              </span>
              <span className="text-gray-700">
                (Probability: {(result.probability * 100).toFixed(1)}%)
              </span>
            </div>
            <div>
              <h3 className="text-md font-medium mb-1">Grad-CAM Visualization</h3>
              <img
                src={`data:image/png;base64,${result.grad_cam}`}
                alt="Grad-CAM"
                className="w-full rounded-lg shadow"
              />
            </div>
          </CardContent>
        )}
      </Card>
    </div>
  );
}

export default App;
