import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload, Image as ImageIcon, Zap, TrendingUp, Award, AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import UserLayout from "@/components/UserLayout";

const Dashboard = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [userContext, setUserContext] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!["image/png", "image/jpeg", "image/jpg"].includes(file.type)) {
        toast({ title: "Invalid Format", description: "Please upload PNG, JPG, or JPEG only", variant: "destructive" });
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        setSelectedImage(reader.result as string);
        setResults(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const mockAnalyze = () => {
    setAnalyzing(true);
    setTimeout(() => {
      const diseases = [
        { name: "Leaf Spot Disease", confidence: 87, severity: "high", emoji: "🔴" },
        { name: "Powdery Mildew", confidence: 72, severity: "medium", emoji: "🟡" },
        { name: "Healthy Plant", confidence: 15, severity: "low", emoji: "🟢" },
      ];
      
      const model1Confidence = 87 + Math.random() * 5;
      const model2Confidence = 83 + Math.random() * 5;
      const bestModel = model1Confidence > model2Confidence ? "Model 1" : "Model 2";
      const topDisease = diseases[0];
      const points = Math.floor(topDisease.confidence * 1.2);

      const result = {
        disease: topDisease.name,
        confidence: topDisease.confidence,
        severity: topDisease.severity,
        emoji: topDisease.emoji,
        model1: { name: "ResNet-50", confidence: model1Confidence.toFixed(1) },
        model2: { name: "EfficientNet-B3", confidence: model2Confidence.toFixed(1) },
        bestModel,
        predictions: diseases,
        points,
        recommendations: {
          symptoms: ["Brown spots on leaves", "Yellowing around affected areas", "Spreading pattern visible"],
          causes: ["Fungal infection", "High humidity", "Poor air circulation"],
          prevention: ["Improve air circulation", "Avoid overhead watering", "Remove infected leaves promptly"],
          treatment: ["Apply fungicide spray", "Increase spacing between plants", "Water at soil level"],
        },
      };

      setResults(result);
      setAnalyzing(false);

      // Update user stats
      const currentUser = JSON.parse(localStorage.getItem("plantdoctor_current_user") || "{}");
      currentUser.points = (currentUser.points || 0) + points;
      currentUser.scans = (currentUser.scans || 0) + 1;
      if (topDisease.confidence >= 90) currentUser.perfectScans = (currentUser.perfectScans || 0) + 1;
      if (topDisease.name !== "Healthy Plant") currentUser.plantsSaved = (currentUser.plantsSaved || 0) + 1;
      localStorage.setItem("plantdoctor_current_user", JSON.stringify(currentUser));

      // Save to history
      const history = JSON.parse(localStorage.getItem("plantdoctor_history") || "[]");
      history.unshift({
        id: Date.now().toString(),
        userId: currentUser.id,
        image: selectedImage,
        userContext,
        result,
        timestamp: new Date().toISOString(),
      });
      localStorage.setItem("plantdoctor_history", JSON.stringify(history.slice(0, 50)));

      toast({
        title: `${topDisease.emoji} Analysis Complete!`,
        description: `Detected: ${topDisease.name} (+${points} points)`,
      });
    }, 2500);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-green-500";
    if (confidence >= 60) return "text-yellow-500";
    return "text-orange-500";
  };

  return (
    <UserLayout>
      <div className="space-y-6 animate-fade-in-up">
        {/* Upload Card */}
        <Card className="glassmorphism border-2 hover-lift p-6">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5 text-primary" />
            Upload Plant Image
          </h3>
          <div className="space-y-4">
            <div
              className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-all hover:bg-muted/50"
              onClick={() => fileInputRef.current?.click()}
            >
              {selectedImage ? (
                <img src={selectedImage} alt="Selected plant" className="max-h-64 mx-auto rounded-lg shadow-md" />
              ) : (
                <div className="space-y-2">
                  <ImageIcon className="w-16 h-16 mx-auto text-muted-foreground" />
                  <p className="text-muted-foreground">Click to upload plant image</p>
                  <p className="text-sm text-muted-foreground">PNG, JPG, JPEG supported</p>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleImageSelect}
                className="hidden"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Additional Context (Optional)</label>
              <Textarea
                placeholder="e.g., 'Yellow spots appeared 3 days ago' or 'Leaves are wilting'"
                value={userContext}
                onChange={(e) => setUserContext(e.target.value)}
                maxLength={200}
                className="resize-none"
              />
              <p className="text-xs text-muted-foreground text-right">{userContext.length}/200</p>
            </div>

            <Button
              className="w-full h-12 text-lg font-semibold rounded-full shadow-md hover:shadow-glow transition-all"
              disabled={!selectedImage || analyzing}
              onClick={mockAnalyze}
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing with AI...
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5 mr-2" />
                  Analyze Plant
                </>
              )}
            </Button>
          </div>
        </Card>

        {/* Results */}
        {results && (
          <div className="space-y-6 animate-scale-in">
            {/* Diagnosis Card */}
            <Card className="glassmorphism border-2 border-primary p-6 hover-lift">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-2xl font-bold flex items-center gap-2">
                    {results.emoji} {results.disease}
                  </h3>
                  <p className="text-muted-foreground">Best Model: {results.bestModel}</p>
                </div>
                <div className="text-right">
                  <div className={`text-3xl font-bold ${getConfidenceColor(results.confidence)}`}>
                    {results.confidence}%
                  </div>
                  <Badge variant="secondary" className="mt-1">
                    +{results.points} Points
                  </Badge>
                </div>
              </div>
              <Progress value={results.confidence} className="h-2" />
            </Card>

            {/* Model Comparison */}
            <Card className="glassmorphism border-2 p-6 hover-lift">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Model Comparison
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{results.model1.name}</span>
                    <span className="font-bold text-primary">{results.model1.confidence}%</span>
                  </div>
                  <Progress value={parseFloat(results.model1.confidence)} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{results.model2.name}</span>
                    <span className="font-bold text-accent">{results.model2.confidence}%</span>
                  </div>
                  <Progress value={parseFloat(results.model2.confidence)} className="h-2" />
                </div>
              </div>
            </Card>

            {/* Top 3 Predictions */}
            <Card className="glassmorphism border-2 p-6 hover-lift">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Award className="w-5 h-5 text-primary" />
                Top 3 Predictions
              </h3>
              <div className="space-y-3">
                {results.predictions.map((pred: any, idx: number) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium flex items-center gap-2">
                        {pred.emoji} {pred.name}
                      </span>
                      <span className={`font-bold ${getConfidenceColor(pred.confidence)}`}>
                        {pred.confidence}%
                      </span>
                    </div>
                    <Progress value={pred.confidence} className="h-1.5" />
                  </div>
                ))}
              </div>
            </Card>

            {/* Recommendations */}
            <Card className="glassmorphism border-2 p-6 hover-lift">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-primary" />
                  Personalized Recommendations
                </h3>
                <Badge variant={results.severity === "high" ? "destructive" : results.severity === "medium" ? "secondary" : "default"}>
                  {results.severity === "high" ? "🔴 Urgent" : results.severity === "medium" ? "🟡 Moderate" : "🟢 Low Priority"}
                </Badge>
              </div>
              
              <Tabs defaultValue="symptoms" className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="symptoms">Symptoms</TabsTrigger>
                  <TabsTrigger value="causes">Causes</TabsTrigger>
                  <TabsTrigger value="prevention">Prevention</TabsTrigger>
                  <TabsTrigger value="treatment">Treatment</TabsTrigger>
                </TabsList>
                <TabsContent value="symptoms" className="space-y-2 mt-4">
                  {results.recommendations.symptoms.map((item: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 mt-1 text-primary shrink-0" />
                      <span>{item}</span>
                    </div>
                  ))}
                </TabsContent>
                <TabsContent value="causes" className="space-y-2 mt-4">
                  {results.recommendations.causes.map((item: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-2">
                      <span className="text-primary">•</span>
                      <span>{item}</span>
                    </div>
                  ))}
                </TabsContent>
                <TabsContent value="prevention" className="space-y-2 mt-4">
                  {results.recommendations.prevention.map((item: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 mt-1 text-green-500 shrink-0" />
                      <span>{item}</span>
                    </div>
                  ))}
                </TabsContent>
                <TabsContent value="treatment" className="space-y-2 mt-4">
                  {results.recommendations.treatment.map((item: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-2">
                      <span className="text-primary font-bold">{idx + 1}.</span>
                      <span>{item}</span>
                    </div>
                  ))}
                </TabsContent>
              </Tabs>
            </Card>
          </div>
        )}

        {/* How It Works */}
        {!results && !analyzing && (
          <div className="grid md:grid-cols-3 gap-4 mt-8">
            {[
              { icon: Upload, title: "Upload Image", desc: "Select a clear photo of your plant" },
              { icon: Zap, title: "AI Analysis", desc: "Dual models analyze the plant health" },
              { icon: CheckCircle2, title: "Get Results", desc: "Receive diagnosis and treatment advice" },
            ].map((step, idx) => (
              <Card key={idx} className="glassmorphism p-6 text-center hover-lift">
                <step.icon className="w-12 h-12 mx-auto mb-3 text-primary" />
                <h4 className="font-bold mb-2">{step.title}</h4>
                <p className="text-sm text-muted-foreground">{step.desc}</p>
              </Card>
            ))}
          </div>
        )}
      </div>
    </UserLayout>
  );
};

export default Dashboard;
