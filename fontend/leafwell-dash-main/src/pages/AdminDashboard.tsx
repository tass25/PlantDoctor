import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Users, Activity, TrendingUp, Target, Clock } from "lucide-react";
import AdminLayout from "@/components/AdminLayout";

const AdminDashboard = () => {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const allUsers = JSON.parse(localStorage.getItem("plantdoctor_users") || "[]");
    const allHistory = JSON.parse(localStorage.getItem("plantdoctor_history") || "[]");
    const regularUsers = allUsers.filter((u: any) => u.role === "user");

    const totalScans = allHistory.length;
    const diseasesFound = allHistory.filter((h: any) => h.result.disease !== "Healthy Plant").length;
    const avgAccuracy = allHistory.reduce((sum: number, h: any) => sum + h.result.confidence, 0) / totalScans || 0;

    // Active users today
    const today = new Date().toDateString();
    const activeToday = new Set(
      allHistory.filter((h: any) => new Date(h.timestamp).toDateString() === today).map((h: any) => h.userId)
    ).size;

    // User growth (last 7 days)
    const userGrowth = [...Array(7)].map((_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const count = regularUsers.filter((u: any) => new Date(u.createdAt) <= date).length;
      return {
        date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        users: count,
      };
    }).reverse();

    // Activity heatmap (last 7 days)
    const activityHeatmap = [...Array(7)].map((_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toDateString();
      const scans = allHistory.filter((h: any) => new Date(h.timestamp).toDateString() === dateStr).length;
      return {
        date: date.toLocaleDateString("en-US", { weekday: "short" }),
        scans,
      };
    }).reverse();

    // Top users by points
    const topUsers = regularUsers
      .sort((a: any, b: any) => (b.points || 0) - (a.points || 0))
      .slice(0, 5)
      .map((u: any) => ({
        name: u.username,
        points: u.points || 0,
      }));

    // Disease distribution
    const diseaseCount: any = {};
    allHistory.forEach((h: any) => {
      const disease = h.result.disease;
      diseaseCount[disease] = (diseaseCount[disease] || 0) + 1;
    });
    const diseaseDistribution = Object.entries(diseaseCount).map(([name, value]) => ({ name, value }));

    // Model performance
    const model1Wins = allHistory.filter((h: any) => h.result.bestModel === "Model 1").length;
    const model2Wins = allHistory.filter((h: any) => h.result.bestModel === "Model 2").length;

    // Accuracy distribution
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

    // User engagement
    const totalBadges = regularUsers.reduce((sum: number, u: any) => sum + (u.badges?.length || 0), 0);
    const avgScansPerUser = totalScans / regularUsers.length || 0;

    // Recent activity
    const recentActivity = allHistory.slice(0, 10).map((h: any) => {
      const user = allUsers.find((u: any) => u.id === h.userId);
      return {
        username: user?.username || "Unknown",
        disease: h.result.disease,
        confidence: h.result.confidence,
        time: new Date(h.timestamp).toLocaleTimeString(),
      };
    });

    // Eco impact
    const ecoImpact = {
      waterSaved: diseasesFound * 15,
      co2Offset: diseasesFound * 2.5,
      pesticideReduced: diseasesFound * 50,
      plantsSaved: diseasesFound,
    };

    setStats({
      totalUsers: regularUsers.length,
      totalScans,
      diseasesFound,
      avgAccuracy: avgAccuracy.toFixed(1),
      activeToday,
      userGrowth,
      activityHeatmap,
      topUsers,
      diseaseDistribution,
      model1Wins,
      model2Wins,
      accuracyRanges,
      totalBadges,
      avgScansPerUser: avgScansPerUser.toFixed(1),
      recentActivity,
      ecoImpact,
    });
  }, []);

  const COLORS = ["#FF4757", "#E84393", "#FFA726", "#AB47BC", "#1ABC9C"];

  if (!stats) {
    return (
      <AdminLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Activity className="w-16 h-16 mx-auto text-primary animate-spin mb-4" />
            <p className="text-muted-foreground">Loading dashboard...</p>
          </div>
        </div>
      </AdminLayout>
    );
  }

  return (
    <AdminLayout>
      <div className="space-y-6 animate-fade-in-up">
        {/* Top Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: "Total Users", value: stats.totalUsers, icon: Users, color: "text-primary" },
            { label: "Total Scans", value: stats.totalScans, icon: Activity, color: "text-accent" },
            { label: "Diseases Found", value: stats.diseasesFound, icon: TrendingUp, color: "text-primary" },
            { label: "System Accuracy", value: `${stats.avgAccuracy}%`, icon: Target, color: "text-accent" },
            { label: "Active Today", value: stats.activeToday, icon: Clock, color: "text-primary" },
          ].map((metric, idx) => (
            <Card key={idx} className="glassmorphism p-4 hover-lift border border-primary/30">
              <metric.icon className={`w-5 h-5 ${metric.color} mb-2`} />
              <div className="text-2xl font-bold animate-counter">{metric.value}</div>
              <div className="text-xs text-muted-foreground">{metric.label}</div>
            </Card>
          ))}
        </div>

        {/* Charts Row 1 */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">User Growth (Last 7 Days)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={stats.userGrowth}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="users" stroke="#FF4757" fill="#FF4757" fillOpacity={0.6} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Activity Heatmap</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={stats.activityHeatmap}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="scans" radius={[8, 8, 0, 0]}>
                  {stats.activityHeatmap.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Top Users by Points</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={stats.topUsers} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" />
                <Tooltip />
                <Bar dataKey="points" fill="#E84393" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift border border-primary/30">
            <h3 className="text-lg font-bold mb-4">Disease Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={stats.diseaseDistribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {stats.diseaseDistribution.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* AI Model Performance */}
        <Card className="glassmorphism p-6 hover-lift border border-primary/30">
          <h3 className="text-xl font-bold mb-4">AI Model Performance</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Model 1 Wins</span>
                  <span className="font-bold text-primary text-xl">{stats.model1Wins}</span>
                </div>
                <Progress value={(stats.model1Wins / stats.totalScans) * 100} className="h-3" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Model 2 Wins</span>
                  <span className="font-bold text-accent text-xl">{stats.model2Wins}</span>
                </div>
                <Progress value={(stats.model2Wins / stats.totalScans) * 100} className="h-3" />
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Accuracy Distribution</h4>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={stats.accuracyRanges}>
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#FF4757" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>

        {/* System Accuracy Gauge */}
        <Card className="glassmorphism p-6 hover-lift border border-primary/30">
          <h3 className="text-xl font-bold mb-4">System Health Metrics</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-medium mb-2">System Accuracy</h4>
              <div className="text-4xl font-bold text-primary mb-2 animate-counter">{stats.avgAccuracy}%</div>
              <Progress value={parseFloat(stats.avgAccuracy)} className="h-3" />
            </div>
            <div>
              <h4 className="font-medium mb-2">Total Badges Earned</h4>
              <div className="text-4xl font-bold text-accent mb-2 animate-counter">{stats.totalBadges}</div>
              <Progress value={(stats.totalBadges / (stats.totalUsers * 8)) * 100} className="h-3" />
            </div>
            <div>
              <h4 className="font-medium mb-2">Avg Scans per User</h4>
              <div className="text-4xl font-bold text-primary mb-2 animate-counter">{stats.avgScansPerUser}</div>
              <Progress value={Math.min((parseFloat(stats.avgScansPerUser) / 10) * 100, 100)} className="h-3" />
            </div>
          </div>
        </Card>

        {/* Eco Impact */}
        <Card className="glassmorphism p-6 hover-lift border-2 border-primary/50">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            🌍 Environmental Impact
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Plants Saved", value: stats.ecoImpact.plantsSaved, emoji: "🌿", unit: "" },
              { label: "Water Saved", value: stats.ecoImpact.waterSaved, emoji: "💧", unit: "L" },
              { label: "CO₂ Offset", value: stats.ecoImpact.co2Offset, emoji: "🌱", unit: "kg" },
              { label: "Pesticide Reduced", value: stats.ecoImpact.pesticideReduced, emoji: "🚫", unit: "ml" },
            ].map((metric, idx) => (
              <div key={idx} className="text-center">
                <div className="text-4xl mb-2">{metric.emoji}</div>
                <div className="text-3xl font-bold text-primary animate-counter">
                  {metric.value.toFixed(1)}{metric.unit}
                </div>
                <div className="text-xs text-muted-foreground">{metric.label}</div>
              </div>
            ))}
          </div>
        </Card>

        {/* Recent Activity Feed */}
        <Card className="glassmorphism p-6 hover-lift border border-primary/30">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Recent Activity (Last 10 Scans)
          </h3>
          <div className="space-y-2">
            {stats.recentActivity.map((activity: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-all">
                <div className="flex items-center gap-3">
                  <Badge variant="outline">{activity.username}</Badge>
                  <span className="text-sm">{activity.disease}</span>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="secondary">{activity.confidence}%</Badge>
                  <span className="text-xs text-muted-foreground">{activity.time}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </AdminLayout>
  );
};

export default AdminDashboard;
