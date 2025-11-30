import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Search, Download, Users, Award, TrendingUp, Activity } from "lucide-react";
import AdminLayout from "@/components/AdminLayout";

const AdminUsers = () => {
  const [users, setUsers] = useState<any[]>([]);
  const [filteredUsers, setFilteredUsers] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedUser, setSelectedUser] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const allUsers = JSON.parse(localStorage.getItem("plantdoctor_users") || "[]");
    const regularUsers = allUsers.filter((u: any) => u.role === "user");
    const allHistory = JSON.parse(localStorage.getItem("plantdoctor_history") || "[]");

    // Enhance users with calculated stats
    const enhancedUsers = regularUsers.map((user: any) => {
      const userScans = allHistory.filter((h: any) => h.userId === user.id);
      const avgAccuracy = userScans.reduce((sum: number, h: any) => sum + h.result.confidence, 0) / userScans.length || 0;
      
      return {
        ...user,
        scans: userScans.length,
        avgAccuracy: avgAccuracy.toFixed(1),
      };
    });

    setUsers(enhancedUsers);
    setFilteredUsers(enhancedUsers);

    const totalBadges = enhancedUsers.reduce((sum: number, u: any) => sum + (u.badges?.length || 0), 0);
    const avgScans = enhancedUsers.reduce((sum: number, u: any) => sum + (u.scans || 0), 0) / enhancedUsers.length || 0;
    const activeUsers = enhancedUsers.filter((u: any) => u.scans > 0).length;

    setStats({
      totalUsers: enhancedUsers.length,
      activeUsers,
      totalBadges,
      avgScans: avgScans.toFixed(1),
    });
  }, []);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    const filtered = users.filter((user) =>
      user.username.toLowerCase().includes(query.toLowerCase())
    );
    setFilteredUsers(filtered);
  };

  const handleUserClick = (user: any) => {
    const allHistory = JSON.parse(localStorage.getItem("plantdoctor_history") || "[]");
    const userScans = allHistory.filter((h: any) => h.userId === user.id);

    // Accuracy over time
    const accuracyData = userScans.slice(0, 10).reverse().map((h: any, idx: number) => ({
      scan: `#${idx + 1}`,
      accuracy: h.result.confidence,
    }));

    // Disease distribution
    const diseaseCount: any = {};
    userScans.forEach((h: any) => {
      const disease = h.result.disease;
      diseaseCount[disease] = (diseaseCount[disease] || 0) + 1;
    });
    const diseaseData = Object.entries(diseaseCount).map(([name, value]) => ({ name, value }));

    // Activity levels (last 7 days)
    const activityData = [...Array(7)].map((_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toDateString();
      const scans = userScans.filter((h: any) => new Date(h.timestamp).toDateString() === dateStr).length;
      return {
        date: date.toLocaleDateString("en-US", { weekday: "short" }),
        scans,
      };
    }).reverse();

    // Badge distribution (mock badges)
    const badgeData = [
      { name: "Earned", value: user.badges?.length || 0 },
      { name: "Remaining", value: 8 - (user.badges?.length || 0) },
    ];

    setSelectedUser({
      ...user,
      accuracyData,
      diseaseData,
      activityData,
      badgeData,
    });
  };

  const exportCSV = () => {
    const csvRows = [
      ["Username", "Join Date", "Points", "Scans", "Badges", "Plants Saved", "Perfect Scans", "Avg Accuracy"],
      ...filteredUsers.map((u) => [
        u.username,
        new Date(u.createdAt).toLocaleDateString(),
        u.points || 0,
        u.scans || 0,
        u.badges?.length || 0,
        u.plantsSaved || 0,
        u.perfectScans || 0,
        u.avgAccuracy || "0",
      ]),
    ];
    const csvStr = csvRows.map((row) => row.join(",")).join("\n");
    const dataBlob = new Blob([csvStr], { type: "text/csv" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `plantdoctor-users-${Date.now()}.csv`;
    link.click();
  };

  const COLORS = ["#FF4757", "#E84393", "#FFA726", "#AB47BC", "#1ABC9C"];

  return (
    <AdminLayout>
      <div className="space-y-6 animate-fade-in-up">
        {/* Top Metrics */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Total Users", value: stats.totalUsers, icon: Users },
              { label: "Active Users", value: stats.activeUsers, icon: Activity },
              { label: "Total Badges", value: stats.totalBadges, icon: Award },
              { label: "Avg Scans/User", value: stats.avgScans, icon: TrendingUp },
            ].map((metric, idx) => (
              <Card key={idx} className="glassmorphism p-4 hover-lift border border-primary/30">
                <metric.icon className="w-5 h-5 text-primary mb-2" />
                <div className="text-2xl font-bold animate-counter">{metric.value}</div>
                <div className="text-xs text-muted-foreground">{metric.label}</div>
              </Card>
            ))}
          </div>
        )}

        {/* Search and Export */}
        <Card className="glassmorphism p-6 border border-primary/30">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search users..."
                  value={searchQuery}
                  onChange={(e) => handleSearch(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Button variant="outline" onClick={exportCSV} className="gap-2 border-primary/50">
              <Download className="w-4 h-4" />
              Export CSV
            </Button>
          </div>
        </Card>

        {/* User Table */}
        <Card className="glassmorphism overflow-hidden border border-primary/30">
          <Table>
            <TableHeader>
              <TableRow className="border-primary/30">
                <TableHead>Username</TableHead>
                <TableHead>Join Date</TableHead>
                <TableHead>Points</TableHead>
                <TableHead>Scans</TableHead>
                <TableHead>Badges</TableHead>
                <TableHead>Plants Saved</TableHead>
                <TableHead>Perfect Scans</TableHead>
                <TableHead>Avg Accuracy</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredUsers.map((user) => (
                <TableRow
                  key={user.id}
                  className="cursor-pointer hover:bg-muted/50 transition-all border-primary/20"
                  onClick={() => handleUserClick(user)}
                >
                  <TableCell className="font-medium">{user.username}</TableCell>
                  <TableCell>{new Date(user.createdAt).toLocaleDateString()}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{user.points || 0}</Badge>
                  </TableCell>
                  <TableCell>{user.scans || 0}</TableCell>
                  <TableCell>{user.badges?.length || 0}</TableCell>
                  <TableCell>{user.plantsSaved || 0}</TableCell>
                  <TableCell>{user.perfectScans || 0}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{user.avgAccuracy}%</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>

        {/* User Details Panel */}
        {selectedUser && (
          <Card className="glassmorphism p-6 border-2 border-primary/50 animate-scale-in">
            <div className="mb-6">
              <h3 className="text-2xl font-bold mb-2">User Details: {selectedUser.username}</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Points", value: selectedUser.points || 0 },
                  { label: "Scans", value: selectedUser.scans || 0 },
                  { label: "Badges", value: selectedUser.badges?.length || 0 },
                  { label: "Plants Saved", value: selectedUser.plantsSaved || 0 },
                ].map((stat, idx) => (
                  <div key={idx} className="text-center p-3 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold text-primary">{stat.value}</div>
                    <div className="text-xs text-muted-foreground">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Accuracy Over Time */}
              <div>
                <h4 className="font-bold mb-3">Accuracy Over Time (Last 10 Scans)</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={selectedUser.accuracyData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="scan" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="accuracy" stroke="#FF4757" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Disease Distribution */}
              <div>
                <h4 className="font-bold mb-3">Disease Distribution</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie data={selectedUser.diseaseData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label>
                      {selectedUser.diseaseData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Activity Levels */}
              <div>
                <h4 className="font-bold mb-3">Activity Levels (Last 7 Days)</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={selectedUser.activityData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="scans" fill="#E84393" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Badge Distribution */}
              <div>
                <h4 className="font-bold mb-3">Badge Distribution</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie data={selectedUser.badgeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label>
                      {selectedUser.badgeData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? "#FF4757" : "#555"} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </Card>
        )}

        {/* Top Performers */}
        <div className="grid md:grid-cols-3 gap-4">
          <Card className="glassmorphism p-4 border border-primary/30">
            <h4 className="font-bold mb-3">Top by Points</h4>
            <div className="space-y-2">
              {users
                .sort((a, b) => (b.points || 0) - (a.points || 0))
                .slice(0, 5)
                .map((user, idx) => (
                  <div key={user.id} className="flex items-center justify-between p-2 rounded bg-muted/50">
                    <span className="text-sm">{idx + 1}. {user.username}</span>
                    <Badge variant="secondary">{user.points || 0}</Badge>
                  </div>
                ))}
            </div>
          </Card>

          <Card className="glassmorphism p-4 border border-primary/30">
            <h4 className="font-bold mb-3">Top by Scans</h4>
            <div className="space-y-2">
              {users
                .sort((a, b) => (b.scans || 0) - (a.scans || 0))
                .slice(0, 5)
                .map((user, idx) => (
                  <div key={user.id} className="flex items-center justify-between p-2 rounded bg-muted/50">
                    <span className="text-sm">{idx + 1}. {user.username}</span>
                    <Badge variant="secondary">{user.scans || 0}</Badge>
                  </div>
                ))}
            </div>
          </Card>

          <Card className="glassmorphism p-4 border border-primary/30">
            <h4 className="font-bold mb-3">Top by Plants Saved</h4>
            <div className="space-y-2">
              {users
                .sort((a, b) => (b.plantsSaved || 0) - (a.plantsSaved || 0))
                .slice(0, 5)
                .map((user, idx) => (
                  <div key={user.id} className="flex items-center justify-between p-2 rounded bg-muted/50">
                    <span className="text-sm">{idx + 1}. {user.username}</span>
                    <Badge variant="secondary">{user.plantsSaved || 0}</Badge>
                  </div>
                ))}
            </div>
          </Card>
        </div>
      </div>
    </AdminLayout>
  );
};

export default AdminUsers;
