import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Shield, LayoutDashboard, Users, BarChart3, LogOut, Menu, X } from "lucide-react";

interface AdminLayoutProps {
  children: React.ReactNode;
}

const AdminLayout = ({ children }: AdminLayoutProps) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [user, setUser] = useState<any>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    document.body.classList.add("admin-theme");
    const currentUser = localStorage.getItem("plantdoctor_current_user");
    if (!currentUser) {
      navigate("/");
    } else {
      const userData = JSON.parse(currentUser);
      if (userData.role !== "admin") {
        navigate("/dashboard");
      }
      setUser(userData);
    }
    return () => {
      document.body.classList.remove("admin-theme");
    };
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("plantdoctor_current_user");
    document.body.classList.remove("admin-theme");
    navigate("/");
  };

  const navItems = [
    { path: "/admin", label: "Dashboard", icon: LayoutDashboard },
    { path: "/admin/users", label: "User Management", icon: Users },
    { path: "/admin/stats", label: "System Stats", icon: BarChart3 },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen admin-theme">
      {/* Top Navigation */}
      <nav className="glassmorphism border-b border-primary/30 sticky top-0 z-50 shadow-glow backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/admin")}>
              <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center shadow-glow animate-pulse-slow">
                <Shield className="w-6 h-6 text-primary-foreground" />
              </div>
              <span className="font-bold text-xl bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                PlantDoctor Admin 👑
              </span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-1">
              {navItems.map((item) => (
                <Button
                  key={item.path}
                  variant={isActive(item.path) ? "default" : "ghost"}
                  onClick={() => navigate(item.path)}
                  className={`gap-2 rounded-full transition-all ${
                    isActive(item.path) ? "shadow-glow" : "hover:bg-muted"
                  }`}
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </Button>
              ))}
              <div className="ml-4 pl-4 border-l border-primary/30">
                <Button
                  variant="outline"
                  onClick={handleLogout}
                  className="gap-2 rounded-full hover:bg-primary hover:text-primary-foreground transition-all border-primary/50"
                >
                  <LogOut className="w-4 h-4" />
                  Logout
                </Button>
              </div>
            </div>

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-primary/30 glassmorphism animate-slide-in-right">
            <div className="px-4 py-4 space-y-2">
              {navItems.map((item) => (
                <Button
                  key={item.path}
                  variant={isActive(item.path) ? "default" : "ghost"}
                  onClick={() => {
                    navigate(item.path);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full justify-start gap-2 ${
                    isActive(item.path) ? "shadow-glow" : ""
                  }`}
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </Button>
              ))}
              <div className="pt-2 border-t border-primary/30">
                <Button
                  variant="outline"
                  onClick={handleLogout}
                  className="w-full gap-2 rounded-full border-primary/50"
                >
                  <LogOut className="w-4 h-4" />
                  Logout
                </Button>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {user && (
          <div className="mb-6 animate-fade-in-up">
            <h2 className="text-2xl font-bold flex items-center gap-2">
              Admin Dashboard <Shield className="w-6 h-6 text-primary" />
            </h2>
            <p className="text-muted-foreground">System monitoring and management</p>
          </div>
        )}
        {children}
      </main>
    </div>
  );
};

export default AdminLayout;
