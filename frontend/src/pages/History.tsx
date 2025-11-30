import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { TrendingUp, Activity, Calendar, Download, ChevronDown, ChevronUp } from "lucide-react";
import UserLayout from "@/components/UserLayout";

const History = () => {
  const [history, setHistory] = useState<any[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const currentUser = JSON.parse(localStorage.getItem("plantdoctor_current_user") || "{}");
    const allHistory = JSON.parse(localStorage.getItem("plantdoctor_history") || "[]");
    const userHistory = allHistory.filter((h: any) => h.userId === currentUser.id);
    setHistory(userHistory);

    // Calculate stats
    const totalScans = userHistory.length;
    const diseases = userHistory.filter((h: any) => h.result.disease !== "Healthy Plant");
    const healthy = userHistory.filter((h: any) => h.result.disease === "Healthy Plant");
    const avgAccuracy = userHistory.reduce((sum: number, h: any) => sum + h.result.confidence, 0) / totalScans || 0;

    const diseaseCount: any = {};
    userHistory.forEach((h: any) => {
      const disease = h.result.disease;
      diseaseCount[disease] = (diseaseCount[disease] || 0) + 1;
    });

    const last7Days = [...Array(7)].map((_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dayScans = userHistory.filter((h: any) => {
        const scanDate = new Date(h.timestamp);
        return scanDate.toDateString() === date.toDateString();
      });
      return {
        date: date.toLocaleDateString("en-US", { weekday: "short" }),
        scans: dayScans.length,
        accuracy: dayScans.reduce((sum: number, h: any) => sum + h.result.confidence, 0) / dayScans.length || 0,
      };
    }).reverse();

    const pieData = Object.entries(diseaseCount).map(([name, value]) => ({ name, value }));

    setStats({
      totalScans,
      plantTypes: new Set(userHistory.map((h: any) => h.result.disease)).size,
      diseasesFound: diseases.length,
      healthyPlants: healthy.length,
      avgAccuracy: avgAccuracy.toFixed(1),
      diseaseData: pieData,
      timelineData: last7Days,
      healthRatio: [
        { name: "Healthy", value: healthy.length, color: "#66BB6A" },
        { name: "Diseased", value: diseases.length, color: "#FF4757" },
      ],
      model1Wins: userHistory.filter((h: any) => h.result.bestModel === "Model 1").length,
      model2Wins: userHistory.filter((h: any) => h.result.bestModel === "Model 2").length,
      perfectScans: userHistory.filter((h: any) => h.result.confidence >= 90).length,
    });
  }, []);

  const COLORS = ["#66BB6A", "#FF4757", "#1ABC9C", "#E84393", "#FFA726", "#AB47BC"];

  const exportData = (format: "json" | "csv") => {
    if (format === "json") {
      const dataStr = JSON.stringify(history, null, 2);
      const dataBlob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `plantdoctor-history-${Date.now()}.json`;
      link.click();
    } else {
      const csvRows = [
        ["Date", "Disease", "Confidence", "Model", "Points"],
        ...history.map((h) => [
          new Date(h.timestamp).toLocaleDateString(),
          h.result.disease,
          h.result.confidence,
          h.result.bestModel,
          h.result.points,
        ]),
      ];
      const csvStr = csvRows.map((row) => row.join(",")).join("\n");
      const dataBlob = new Blob([csvStr], { type: "text/csv" });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `plantdoctor-history-${Date.now()}.csv`;
      link.click();
    }
  };

  return (
    <UserLayout>
      <div className="space-y-6 animate-fade-in-up">
        {/* Top Metrics */}
        {stats && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { label: "Total Analyses", value: stats.totalScans, icon: Activity },
                { label: "Plant Types", value: stats.plantTypes, icon: TrendingUp },
                { label: "Diseases Found", value: stats.diseasesFound, icon: Activity },
                { label: "Healthy Plants", value: stats.healthyPlants, icon: Activity },
                { label: "Avg Accuracy", value: `${stats.avgAccuracy}%`, icon: TrendingUp },
              ].map((metric, idx) => (
                <Card key={idx} className="glassmorphism p-4 hover-lift">
                  <metric.icon className="w-5 h-5 text-primary mb-2" />
                  <div className="text-2xl font-bold animate-counter">{metric.value}</div>
                  <div className="text-xs text-muted-foreground">{metric.label}</div>
                </Card>
              ))}
            </div>

            {/* Charts */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Disease Distribution */}
              <Card className="glassmorphism p-6 hover-lift">
                <h3 className="text-lg font-bold mb-4">Disease Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={stats.diseaseData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                      {stats.diseaseData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Card>

              {/* Health vs Disease */}
              <Card className="glassmorphism p-6 hover-lift">
                <h3 className="text-lg font-bold mb-4">Health vs Disease Ratio</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={stats.healthRatio}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {stats.healthRatio.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              {/* Accuracy Timeline */}
              <Card className="glassmorphism p-6 hover-lift">
                <h3 className="text-lg font-bold mb-4">Accuracy Timeline (7 Days)</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={stats.timelineData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#66BB6A" strokeWidth={3} dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              {/* Activity Heatmap */}
              <Card className="glassmorphism p-6 hover-lift">
                <h3 className="text-lg font-bold mb-4">Daily Activity</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={stats.timelineData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="scans" fill="#1ABC9C" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Performance Gauges */}
            <div className="grid md:grid-cols-3 gap-4">
              <Card className="glassmorphism p-4 hover-lift">
                <h4 className="font-bold mb-2">Average Accuracy</h4>
                <div className="text-3xl font-bold text-primary mb-2">{stats.avgAccuracy}%</div>
                <Progress value={parseFloat(stats.avgAccuracy)} className="h-2" />
              </Card>
              <Card className="glassmorphism p-4 hover-lift">
                <h4 className="font-bold mb-2">Disease Detection Rate</h4>
                <div className="text-3xl font-bold text-primary mb-2">
                  {((stats.diseasesFound / stats.totalScans) * 100).toFixed(0)}%
                </div>
                <Progress value={(stats.diseasesFound / stats.totalScans) * 100} className="h-2" />
              </Card>
              <Card className="glassmorphism p-4 hover-lift">
                <h4 className="font-bold mb-2">Perfect Scan Rate</h4>
                <div className="text-3xl font-bold text-primary mb-2">
                  {((stats.perfectScans / stats.totalScans) * 100).toFixed(0)}%
                </div>
                <Progress value={(stats.perfectScans / stats.totalScans) * 100} className="h-2" />
              </Card>
            </div>

            {/* Model Performance */}
            <Card className="glassmorphism p-6 hover-lift">
              <h3 className="text-lg font-bold mb-4">Model Performance</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">Model 1 Wins</span>
                    <span className="font-bold text-primary">{stats.model1Wins}</span>
                  </div>
                  <Progress value={(stats.model1Wins / stats.totalScans) * 100} className="h-2" />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">Model 2 Wins</span>
                    <span className="font-bold text-accent">{stats.model2Wins}</span>
                  </div>
                  <Progress value={(stats.model2Wins / stats.totalScans) * 100} className="h-2" />
                </div>
              </div>
            </Card>

            {/* Eco Metrics */}
            <Card className="glassmorphism p-6 hover-lift border-2 border-primary/30">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                🌍 Environmental Impact
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Plants Saved", value: stats.diseasesFound, emoji: "🌿" },
                  { label: "Water Saved (L)", value: stats.diseasesFound * 15, emoji: "💧" },
                  { label: "CO₂ Offset (kg)", value: stats.diseasesFound * 2.5, emoji: "🌱" },
                  { label: "Pesticide Reduced (ml)", value: stats.diseasesFound * 50, emoji: "🚫" },
                ].map((metric, idx) => (
                  <div key={idx} className="text-center">
                    <div className="text-3xl mb-1">{metric.emoji}</div>
                    <div className="text-2xl font-bold text-primary animate-counter">{metric.value.toFixed(1)}</div>
                    <div className="text-xs text-muted-foreground">{metric.label}</div>
                  </div>
                ))}
              </div>
            </Card>
          </>
        )}

        {/* History List */}
        <Card className="glassmorphism p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold flex items-center gap-2">
              <Calendar className="w-5 h-5 text-primary" />
              Analysis History
            </h3>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => exportData("json")} className="gap-2">
                <Download className="w-4 h-4" />
                JSON
              </Button>
              <Button variant="outline" size="sm" onClick={() => exportData("csv")} className="gap-2">
                <Download className="w-4 h-4" />
                CSV
              </Button>
            </div>
          </div>

          {history.length === 0 ? (
            <div className="text-center py-12">
              <Activity className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No analyses yet. Upload a plant image to start!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((item) => (
                <Card
                  key={item.id}
                  className="glassmorphism p-4 hover-lift cursor-pointer"
                  onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}
                >
                  <div className="flex items-start gap-4">
                    <img src={item.image} alt="Plant" className="w-20 h-20 rounded-lg object-cover" />
                    <div className="flex-1">
                      <div className="flex items-start justify-between">
                        <div>
                          <h4 className="font-bold flex items-center gap-2">
                            {item.result.emoji} {item.result.disease}
                          </h4>
                          <p className="text-sm text-muted-foreground">
                            {new Date(item.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <div className="text-right">
                          <Badge variant="secondary" className="font-bold">
                            {item.result.confidence}%
                          </Badge>
                          <p className="text-xs text-muted-foreground mt-1">+{item.result.points} pts</p>
                        </div>
                      </div>
                      {expandedId === item.id && (
                        <div className="mt-4 space-y-3 animate-fade-in-up">
                          {item.userContext && (
                            <div className="bg-muted/50 p-3 rounded-lg">
                              <p className="text-sm"><strong>Context:</strong> {item.userContext}</p>
                            </div>
                          )}
                          <div className="grid md:grid-cols-2 gap-3">
                            <div>
                              <h5 className="font-semibold text-sm mb-2">Model Comparison</h5>
                              <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                  <span>{item.result.model1.name}</span>
                                  <span className="font-bold">{item.result.model1.confidence}%</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                  <span>{item.result.model2.name}</span>
                                  <span className="font-bold">{item.result.model2.confidence}%</span>
                                </div>
                              </div>
                            </div>
                            <div>
                              <h5 className="font-semibold text-sm mb-2">Top Predictions</h5>
                              {item.result.predictions.slice(0, 2).map((pred: any, idx: number) => (
                                <div key={idx} className="text-sm flex justify-between">
                                  <span>{pred.emoji} {pred.name}</span>
                                  <span className="font-bold">{pred.confidence}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                    {expandedId === item.id ? (
                      <ChevronUp className="w-5 h-5 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-muted-foreground" />
                    )}
                  </div>
                </Card>
              ))}
            </div>
          )}
        </Card>
      </div>
    </UserLayout>
  );
};

export default History;
