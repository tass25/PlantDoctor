import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Trophy, Award, Star, Crown, Medal, Target } from "lucide-react";
import { BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend, Cell, PieChart, Pie } from "recharts";
import UserLayout from "@/components/UserLayout";

const BADGES = [
  { id: "first_scan", name: "First Steps", emoji: "🌱", requirement: 1, desc: "Complete your first scan" },
  { id: "novice", name: "Plant Novice", emoji: "🌿", requirement: 5, desc: "Complete 5 scans" },
  { id: "expert", name: "Plant Expert", emoji: "🌳", requirement: 20, desc: "Complete 20 scans" },
  { id: "master", name: "Plant Master", emoji: "🏆", requirement: 50, desc: "Complete 50 scans" },
  { id: "savior", name: "Plant Savior", emoji: "💚", requirement: 10, desc: "Save 10 plants" },
  { id: "perfect", name: "Perfect Eye", emoji: "👁️", requirement: 5, desc: "Get 5 perfect scans (90%+)" },
  { id: "eco_warrior", name: "Eco Warrior", emoji: "🌍", requirement: 25, desc: "Save 25 plants" },
  { id: "legend", name: "Plant Legend", emoji: "🌟", requirement: 100, desc: "Complete 100 scans" },
];

const Leaderboard = () => {
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [userBadges, setUserBadges] = useState<any[]>([]);

  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("plantdoctor_current_user") || "{}");
    const allUsers = JSON.parse(localStorage.getItem("plantdoctor_users") || "[]");
    
    // Sort users by points
    const sorted = allUsers
      .filter((u: any) => u.role === "user")
      .sort((a: any, b: any) => (b.points || 0) - (a.points || 0))
      .slice(0, 10);

    setCurrentUser(user);
    setLeaderboard(sorted);

    // Calculate earned badges
    const earned = BADGES.map((badge) => {
      let progress = 0;
      let isEarned = false;
      let earnedDate = null;

      switch (badge.id) {
        case "first_scan":
        case "novice":
        case "expert":
        case "master":
        case "legend":
          progress = (user.scans || 0) / badge.requirement;
          isEarned = (user.scans || 0) >= badge.requirement;
          break;
        case "savior":
        case "eco_warrior":
          progress = (user.plantsSaved || 0) / badge.requirement;
          isEarned = (user.plantsSaved || 0) >= badge.requirement;
          break;
        case "perfect":
          progress = (user.perfectScans || 0) / badge.requirement;
          isEarned = (user.perfectScans || 0) >= badge.requirement;
          break;
      }

      if (isEarned && !earnedDate) {
        earnedDate = new Date().toLocaleDateString();
      }

      return { ...badge, progress: Math.min(progress * 100, 100), isEarned, earnedDate };
    });

    setUserBadges(earned);
  }, []);

  const COLORS = ["#66BB6A", "#1ABC9C", "#FFA726", "#AB47BC", "#FF4757"];

  const topPointsData = leaderboard.slice(0, 5).map((u) => ({
    name: u.username,
    points: u.points || 0,
  }));

  const topSaviorsData = leaderboard
    .sort((a, b) => (b.plantsSaved || 0) - (a.plantsSaved || 0))
    .slice(0, 5)
    .map((u) => ({
      name: u.username,
      saved: u.plantsSaved || 0,
    }));

  const badgeDistribution = BADGES.map((badge) => ({
    name: badge.name,
    value: leaderboard.filter((u) => {
      switch (badge.id) {
        case "first_scan":
        case "novice":
        case "expert":
        case "master":
        case "legend":
          return (u.scans || 0) >= badge.requirement;
        case "savior":
        case "eco_warrior":
          return (u.plantsSaved || 0) >= badge.requirement;
        case "perfect":
          return (u.perfectScans || 0) >= badge.requirement;
        default:
          return false;
      }
    }).length,
  }));

  const radarData = [
    { stat: "Points", user: currentUser?.points || 0, max: Math.max(...leaderboard.map((u) => u.points || 0), 100) },
    { stat: "Scans", user: currentUser?.scans || 0, max: Math.max(...leaderboard.map((u) => u.scans || 0), 10) },
    { stat: "Plants Saved", user: currentUser?.plantsSaved || 0, max: Math.max(...leaderboard.map((u) => u.plantsSaved || 0), 10) },
    { stat: "Badges", user: userBadges.filter((b) => b.isEarned).length, max: BADGES.length },
    { stat: "Perfect Scans", user: currentUser?.perfectScans || 0, max: Math.max(...leaderboard.map((u) => u.perfectScans || 0), 5) },
  ].map((item) => ({
    ...item,
    userPercent: (item.user / item.max) * 100,
    maxPercent: 100,
  }));

  const currentUserRank = leaderboard.findIndex((u) => u.id === currentUser?.id) + 1;

  return (
    <UserLayout>
      <div className="space-y-6 animate-fade-in-up">
        {/* User Profile Card */}
        <Card className="glassmorphism border-2 border-primary p-6 hover-lift">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-primary rounded-full flex items-center justify-center text-3xl shadow-glow">
                {currentUser?.username?.[0]?.toUpperCase() || "?"}
              </div>
              <div>
                <h2 className="text-2xl font-bold">{currentUser?.username}</h2>
                <p className="text-muted-foreground">
                  Rank #{currentUserRank > 0 ? currentUserRank : "Unranked"}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "Total Points", value: currentUser?.points || 0, icon: Star },
                { label: "Total Scans", value: currentUser?.scans || 0, icon: Target },
                { label: "Plants Saved", value: currentUser?.plantsSaved || 0, icon: Trophy },
                { label: "Perfect Scans", value: currentUser?.perfectScans || 0, icon: Medal },
              ].map((stat, idx) => (
                <div key={idx} className="text-center">
                  <stat.icon className="w-5 h-5 mx-auto text-primary mb-1" />
                  <div className="text-2xl font-bold animate-counter">{stat.value}</div>
                  <div className="text-xs text-muted-foreground">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Badge Collection */}
        <Card className="glassmorphism p-6 hover-lift">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Award className="w-5 h-5 text-primary" />
            Badge Collection
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {userBadges.map((badge) => (
              <Card
                key={badge.id}
                className={`p-4 text-center transition-all ${
                  badge.isEarned
                    ? "glassmorphism border-2 border-primary shadow-glow badge-glow"
                    : "opacity-50 bg-muted"
                }`}
              >
                <div className="text-4xl mb-2">{badge.emoji}</div>
                <h4 className="font-bold text-sm mb-1">{badge.name}</h4>
                <p className="text-xs text-muted-foreground mb-2">{badge.desc}</p>
                {badge.isEarned ? (
                  <Badge variant="default" className="text-xs">
                    Unlocked {badge.earnedDate}
                  </Badge>
                ) : (
                  <div className="space-y-1">
                    <Progress value={badge.progress} className="h-1.5" />
                    <p className="text-xs text-muted-foreground">{badge.progress.toFixed(0)}%</p>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </Card>

        {/* Global Leaderboard */}
        <Card className="glassmorphism p-6 hover-lift">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Trophy className="w-5 h-5 text-primary" />
            Global Leaderboard
          </h3>
          <div className="space-y-2">
            {leaderboard.map((user, idx) => (
              <Card
                key={user.id}
                className={`glassmorphism p-4 flex items-center gap-4 transition-all ${
                  user.id === currentUser?.id ? "border-2 border-primary shadow-glow" : ""
                }`}
              >
                <div className="flex-shrink-0">
                  {idx === 0 && <Crown className="w-8 h-8 text-yellow-500" />}
                  {idx === 1 && <Medal className="w-8 h-8 text-gray-400" />}
                  {idx === 2 && <Medal className="w-8 h-8 text-amber-700" />}
                  {idx > 2 && <span className="text-2xl font-bold text-muted-foreground">#{idx + 1}</span>}
                </div>
                <div className="flex-1 grid grid-cols-2 md:grid-cols-6 gap-2 items-center">
                  <div className="font-bold">{user.username}</div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Points:</span>{" "}
                    <span className="font-bold text-primary">{user.points || 0}</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Scans:</span> {user.scans || 0}
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Saved:</span> {user.plantsSaved || 0}
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Badges:</span>{" "}
                    {
                      BADGES.filter((badge) => {
                        switch (badge.id) {
                          case "first_scan":
                          case "novice":
                          case "expert":
                          case "master":
                          case "legend":
                            return (user.scans || 0) >= badge.requirement;
                          case "savior":
                          case "eco_warrior":
                            return (user.plantsSaved || 0) >= badge.requirement;
                          case "perfect":
                            return (user.perfectScans || 0) >= badge.requirement;
                          default:
                            return false;
                        }
                      }).length
                    }
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Perfect:</span> {user.perfectScans || 0}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="glassmorphism p-6 hover-lift">
            <h3 className="text-lg font-bold mb-4">Top Points Leaders</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={topPointsData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="points" radius={[8, 8, 0, 0]}>
                  {topPointsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift">
            <h3 className="text-lg font-bold mb-4">Top Plant Saviors</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={topSaviorsData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="saved" fill="#66BB6A" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift">
            <h3 className="text-lg font-bold mb-4">Badge Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={badgeDistribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {badgeDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card className="glassmorphism p-6 hover-lift">
            <h3 className="text-lg font-bold mb-4">You vs Top Performers</h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="stat" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="You" dataKey="userPercent" stroke="#66BB6A" fill="#66BB6A" fillOpacity={0.6} />
                <Radar name="Max" dataKey="maxPercent" stroke="#FF4757" fill="#FF4757" fillOpacity={0.3} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Achievement Progress */}
        <Card className="glassmorphism p-6 hover-lift">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            Achievement Progress
          </h3>
          <div className="space-y-4">
            {userBadges.filter((b) => !b.isEarned).map((badge) => (
              <div key={badge.id} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{badge.emoji}</span>
                    <div>
                      <h4 className="font-bold">{badge.name}</h4>
                      <p className="text-sm text-muted-foreground">{badge.desc}</p>
                    </div>
                  </div>
                  <Badge variant="outline">{badge.progress.toFixed(0)}%</Badge>
                </div>
                <Progress value={badge.progress} className="h-2" />
              </div>
            ))}
          </div>
        </Card>
      </div>
    </UserLayout>
  );
};

export default Leaderboard;
