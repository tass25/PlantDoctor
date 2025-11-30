import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { TrendingUp, Award, Activity, Target, Clock } from "lucide-react";
import AdminLayout from "@/components/AdminLayout";

const AdminStats = () => {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const allUsers = JSON.parse(localStorage.getItem("plantdoctor_users") || "[]");
    const allHistory = JSON.parse(localStorage.getItem("plantdoctor_history") || "[]");
    const regularUsers = allUsers.filter((u: any) => u.role === "user");

    // Badge distribution
    const BADGES = [
      { id: "first_scan", name: "First Steps", requirement: 1, type: "scans" },
      { id: "novice", name: "Plant Novice", requirement: 5, type: "scans" },
      { id: "expert", name: "Plant Expert", requirement: 20, type: "scans" },
      { id: "master", name: "Plant Master", requirement: 50, type: "scans" },
      { id: "savior", name: "Plant Savior", requirement: 10, type: "plantsSaved" },
      { id: "perfect", name: "Perfect Eye", requirement: 5, type: "perfectScans" },
      { id: "eco_warrior", name: "Eco Warrior", requirement: 25, type: "plantsSaved" },
      { id: "legend", name: "Plant Legend", requirement: 100, type: "scans" },
    ];

    const badgeDistribution = BADGES.map((badge) => ({
      name: badge.name,
      value: regularUsers.filter((u: any) => (u[badge.type] || 0) >= badge.requirement).length,
    }));

    // Points distribution
    const pointsRanges = [
      { range: "0-100", count: 0 },
      { range: "100-500", count: 0 },
      { range: "500-1000", count: 0 },
      { range: "1000-2000", count: 0 },
      { range: "2000+", count: 0 },
    ];
    regularUsers.forEach((u: any) => {
      const points = u.points || 0;
      if (points < 100) pointsRanges[0].count++;
      else if (points < 500) pointsRanges[1].count++;
      else if (points < 1000) pointsRanges[2].count++;
      else if (points < 2000) pointsRanges[3].count++;
      else pointsRanges[4].count++;
    });

    // Disease frequency & severity
    const diseaseStats: any = {};
    allHistory.forEach((h: any) => {
      const disease = h.result.disease;
      if (!diseaseStats[disease]) {
        diseaseStats[disease] = { count: 0, totalSeverity: 0 };
      }
      diseaseStats[disease].count++;
      const severityScore = h.result.severity === "high" ? 3 : h.result.severity === "medium" ? 2 : 1;
      diseaseStats[disease].totalSeverity += severityScore;
    });

    const diseaseData = Object.entries(diseaseStats)
      .map(([name, data]: [string, any]) => ({
        name,
        frequency: data.count,
        avgSeverity: (data.totalSeverity / data.count).toFixed(1),
      }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10);

    // Model performance
    const model1Wins = allHistory.filter((h: any) => h.result.bestModel === "Model 1").length;
    const model2Wins = allHistory.filter((h: any) => h.result.bestModel === "Model 2").length;

    const model1Accuracies = allHistory
      .filter((h: any) => h.result.bestModel === "Model 1")
      .map((h: any) => parseFloat(h.result.model1.confidence));
    const model2Accuracies = allHistory
      .filter((h: any) => h.result.bestModel === "Model 2")
      .map((h: any) => parseFloat(h.result.model2.confidence));

    const model1AvgAccuracy = model1Accuracies.reduce((a, b) => a + b, 0) / model1Accuracies.length || 0;
    const model2AvgAccuracy = model2Accuracies.reduce((a, b) => a + b, 0) / model2Accuracies.length || 0;

    // Accuracy ranges
    const accuracyRanges = [
      { range: "0-50%", count: 0 },
      { range: "50-70%", count: 0 },
      { range: "70-85%", count: 0 },
      { range: "85-95%", count: 0 },
      { range: "95-100%", count: 0 },
    ];
    allHistory.forEach((h: any) => {
      const conf = h.result.confidence;
      if (conf < 50) accuracyRanges[0].count++;
      else if (conf < 70) accuracyRanges[1].count++;
      else if (conf < 85) accuracyRanges[2].count++;
      else if (conf < 95) accuracyRanges[3].count++;
      else accuracyRanges[4].count++;
    });

    // Weekly activity trends
    const weeklyTrends = [...Array(7)].map((_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toDateString();
      const dayScans = allHistory.filter((h: any) => new Date(h.timestamp).toDateString() === dateStr);
      return {
        day: date.toLocaleDateString("en-US", { weekday: "short" }),
        scans: dayScans.length,
        avgAccuracy: dayScans.reduce((sum: number, h: any) => sum + h.result.confidence, 0) / dayScans.length || 0,
      };
    }).reverse();

    // Peak usage hours (0-23)
    const hourlyUsage = Array.from({ length: 24 }, (_, hour) => ({
      hour: `${hour}:00`,
      scans: allHistory.filter((h: any) => new Date(h.timestamp).getHours() === hour).length,
    }));

    // Environmental impact
    const diseasesFound = allHistory.filter((h: any) => h.result.disease !== "Healthy Plant").length;
    const ecoImpact = {
      waterSaved: diseasesFound * 15,
      co2Offset: diseasesFound * 2.5,
      pesticideReduced: diseasesFound * 50,
      plantsSaved: diseasesFound,
    };

    // Health gauges
    const totalScans = allHistory.length;
    const avgAccuracy = allHistory.reduce((sum: number, h: any) => sum + h.result.confidence, 0) / totalScans || 0;
    const activeUsers = regularUsers.filter((u: any) => (u.scans || 0) > 0).length;
    const retentionRate = (activeUsers / regularUsers.length) * 100 || 0;
    const engagementScore = (totalScans / regularUsers.length) * 10 || 0;

    setStats({
      badgeDistribution,
      pointsRanges,
      diseaseData,
      model1Wins,
      model2Wins,
      model1AvgAccuracy: model1AvgAccuracy.toFixed(1),
      model2AvgAccuracy: model2AvgAccuracy.toFixed(1),
      accuracyRanges,
      weeklyTrends,
      hourlyUsage,
      ecoImpact,
      retentionRate: retentionRate.toFixed(1),
      systemAccuracy: avgAccuracy.toFixed(1),
      engagementScore: Math.min(engagementScore, 100).toFixed(1),
    });
  }, []);

  const COLORS = ["#FF4757", "#E84393", "#FFA726", "#AB47BC", "#1ABC9C", "#00D2FF"];

  if (!stats) {
    return (
      <AdminLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Activity className="w-16 h-16 mx-auto text-primary animate-spin mb-4" />
            <p className="text-muted-foreground">Loading statistics...</p>
          </div>
        </div>
      </AdminLayout>
    );
  }

  return (
    <AdminLayout>
      <div className="space-y-6 animate-fade-in-up">
        {/* Badge Distribution */}
        <Card className="glassmorphism p-6 hover-lift border border-primary/30">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Award className="w-5 h-5 text-primary" />
            Badge Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stats.badgeDistribution}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                {stats.badgeDistribution.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Points & Disease */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Points Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={stats.pointsRanges} dataKey="count" nameKey="range" cx="50%" cy="50%" outerRadius={80} label>
                  {stats.pointsRanges.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Top 10 Disease Frequency</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={stats.diseaseData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip />
                <Bar dataKey="frequency" fill="#FF4757" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* AI Model Performance */}
        <Card className="glassmorphism p-6 hover-lift border border-primary/30">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            AI Model Performance Comparison
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Win Rate</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Model 1</span>
                    <Badge variant="secondary">{stats.model1Wins} wins</Badge>
                  </div>
                  <Progress value={(stats.model1Wins / (stats.model1Wins + stats.model2Wins)) * 100} className="h-3" />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Model 2</span>
                    <Badge variant="secondary">{stats.model2Wins} wins</Badge>
                  </div>
                  <Progress value={(stats.model2Wins / (stats.model1Wins + stats.model2Wins)) * 100} className="h-3" />
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Average Accuracy</h4>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary mb-1">{stats.model1AvgAccuracy}%</div>
                  <div className="text-sm text-muted-foreground">Model 1</div>
                  <Progress value={parseFloat(stats.model1AvgAccuracy)} className="h-2 mt-2" />
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-accent mb-1">{stats.model2AvgAccuracy}%</div>
                  <div className="text-sm text-muted-foreground">Model 2</div>
                  <Progress value={parseFloat(stats.model2AvgAccuracy)} className="h-2 mt-2" />
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">Accuracy Distribution</h4>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={stats.accuracyRanges}>
                  <XAxis dataKey="range" angle={-45} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#E84393" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>

        {/* Weekly Activity & Peak Hours */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Weekly Activity Trends</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={stats.weeklyTrends}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="day" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="scans" stroke="#FF4757" strokeWidth={2} dot={{ r: 5 }} />
                <Line yAxisId="right" type="monotone" dataKey="avgAccuracy" stroke="#1ABC9C" strokeWidth={2} dot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Peak Usage Hours</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={stats.hourlyUsage}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="hour" angle={-45} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="scans" radius={[8, 8, 0, 0]}>
                  {stats.hourlyUsage.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Environmental Impact */}
        <Card className="glassmorphism p-6 hover-lift border-2 border-primary/50">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            🌍 Total Environmental Impact
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Plants Saved", value: stats.ecoImpact.plantsSaved, emoji: "🌿", unit: "" },
              { label: "Water Saved", value: stats.ecoImpact.waterSaved, emoji: "💧", unit: "L" },
              { label: "CO₂ Offset", value: stats.ecoImpact.co2Offset, emoji: "🌱", unit: "kg" },
              { label: "Pesticide Reduced", value: stats.ecoImpact.pesticideReduced, emoji: "🚫", unit: "ml" },
            ].map((metric, idx) => (
              <div key={idx} className="text-center p-4 bg-muted/30 rounded-lg">
                <div className="text-5xl mb-2">{metric.emoji}</div>
                <div className="text-4xl font-bold text-primary animate-counter">
                  {metric.value.toFixed(1)}{metric.unit}
                </div>
                <div className="text-sm text-muted-foreground mt-1">{metric.label}</div>
              </div>
            ))}
          </div>
        </Card>

        {/* Health Gauges */}
        <Card className="glassmorphism p-6 hover-lift border border-primary/30">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            System Health Indicators
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium mb-2">Retention Rate</h4>
              <div className="text-4xl font-bold text-primary mb-2 animate-counter">{stats.retentionRate}%</div>
              <Progress value={parseFloat(stats.retentionRate)} className="h-3" />
              <p className="text-xs text-muted-foreground mt-2">Active users / Total users</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">System Accuracy</h4>
              <div className="text-4xl font-bold text-accent mb-2 animate-counter">{stats.systemAccuracy}%</div>
              <Progress value={parseFloat(stats.systemAccuracy)} className="h-3" />
              <p className="text-xs text-muted-foreground mt-2">Average detection accuracy</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">Engagement Score</h4>
              <div className="text-4xl font-bold text-primary mb-2 animate-counter">{stats.engagementScore}%</div>
              <Progress value={parseFloat(stats.engagementScore)} className="h-3" />
              <p className="text-xs text-muted-foreground mt-2">User activity level</p>
            </div>
          </div>
        </Card>
      </div>
    </AdminLayout>
  );
};

export default AdminStats;
