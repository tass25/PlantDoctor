import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";

export default function Dashboard() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => setSelectedImage(reader.result as string);
    reader.readAsDataURL(file);
  };

  const analyze = async () => {
    if (!selectedImage) return;

    setAnalyzing(true);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: selectedImage }),
      });

      const data = await res.json();

      setResults({
        disease: data.disease,
        confidence: data.confidence, // already percentage from backend
        severity: data.severity,
        emoji: data.emoji,
        bestModel: data.bestModel,
        predictions: data.predictions.map((p: any) => ({
          name: p.name,
          confidence: Math.round(p.confidence * 100),
          emoji: p.emoji,
        })),
        recommendations: data.recommendations,
      });
    } catch (err) {
      console.error(err);
      toast({
        title: "Error",
        description: "Prediction failed",
        variant: "destructive",
      });
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Plant Disease Analyzer</CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          <input type="file" accept="image/*" onChange={handleImageUpload} />

          {selectedImage && (
            <img
              src={selectedImage}
              alt="Selected"
              className="w-64 mt-4 rounded shadow"
            />
          )}

          <Button onClick={analyze} disabled={analyzing || !selectedImage}>
            {analyzing ? "Analyzing..." : "Analyze"}
          </Button>
        </CardContent>
      </Card>

      {results && (
        <Card>
          <CardHeader>
            <CardTitle>
              {results.emoji} {results.disease} ({results.confidence}%)
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            <p>
              <strong>Severity:</strong> {results.severity}
            </p>

            <p>
              <strong>Best Model:</strong> {results.bestModel}
            </p>

            <div>
              <strong>Top Predictions:</strong>
              <ul className="list-disc ml-6">
                {results.predictions.map((p: any, idx: number) => (
                  <li key={idx}>
                    {p.emoji} {p.name} — {p.confidence}%
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <strong>Recommendations:</strong>
              <ul className="list-disc ml-6">
                {results.recommendations.symptoms.map((s: string, i: number) => (
                  <li key={i}>Symptom: {s}</li>
                ))}
                {results.recommendations.causes.map((c: string, i: number) => (
                  <li key={i}>Cause: {c}</li>
                ))}
                {results.recommendations.prevention.map((p: string, i: number) => (
                  <li key={i}>Prevention: {p}</li>
                ))}
                {results.recommendations.treatment.map((t: string, i: number) => (
                  <li key={i}>Treatment: {t}</li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
