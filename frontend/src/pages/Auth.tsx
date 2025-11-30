import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card } from "@/components/ui/card";
import { Leaf, Sun, Moon } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { authAPI } from "../services/api";

const Auth = () => {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [registerUsername, setRegisterUsername] = useState("");
  const [registerPassword, setRegisterPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "dark");
  };

  const handleLogin = async () => {
    if (!loginUsername || !loginPassword) {
      toast({ title: "Error", description: "Please fill all fields", variant: "destructive" });
      return;
    }
    
    setLoading(true);
    try {
      const response = await authAPI.login(loginUsername, loginPassword);
      
      // Store user data with token
      const userData = {
        ...response.user,
        access_token: response.access_token,
      };
      localStorage.setItem("plantdoctor_current_user", JSON.stringify(userData));
      
      toast({ title: "Welcome back! 🌿", description: "Login successful" });
      navigate(response.user.role === "admin" ? "/admin" : "/dashboard");
    } catch (error: any) {
      toast({ 
        title: "Error", 
        description: error.message || "Login failed", 
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    if (!registerUsername || !registerPassword || !confirmPassword) {
      toast({ title: "Error", description: "Please fill all fields", variant: "destructive" });
      return;
    }
    if (registerUsername.length < 3) {
      toast({ title: "Error", description: "Username must be at least 3 characters", variant: "destructive" });
      return;
    }
    if (registerPassword.length < 6) {
      toast({ title: "Error", description: "Password must be at least 6 characters", variant: "destructive" });
      return;
    }
    if (registerPassword !== confirmPassword) {
      toast({ title: "Error", description: "Passwords do not match", variant: "destructive" });
      return;
    }

    setLoading(true);
    try {
      const response = await authAPI.register(registerUsername, registerPassword, confirmPassword);
      
      // Store user data with token
      const userData = {
        ...response.user,
        access_token: response.access_token,
      };
      localStorage.setItem("plantdoctor_current_user", JSON.stringify(userData));
      
      toast({ title: "Welcome! 🌱", description: "Account created successfully" });
      navigate("/dashboard");
    } catch (error: any) {
      toast({ 
        title: "Error", 
        description: error.message || "Registration failed", 
        variant: "destructive" 
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 opacity-20">
        {[...Array(10)].map((_, i) => (
          <Leaf
            key={i}
            className="absolute text-primary animate-bounce-subtle"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.5}s`,
              fontSize: `${20 + Math.random() * 30}px`,
            }}
          />
        ))}
      </div>

      <Card className="w-full max-w-md glassmorphism border-2 shadow-card relative z-10 animate-scale-in">
        <div className="p-8">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-primary rounded-full mb-4 shadow-glow">
              <Leaf className="w-8 h-8 text-primary-foreground" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              PlantDoctor
            </h1>
            <p className="text-muted-foreground mt-2">AI-Powered Plant Health Analysis</p>
          </div>

          <Tabs defaultValue="login" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="login" className="transition-all">Login</TabsTrigger>
              <TabsTrigger value="register" className="transition-all">Register</TabsTrigger>
            </TabsList>

            <TabsContent value="login" className="space-y-4 animate-fade-in-up">
              <div className="space-y-2">
                <Label htmlFor="login-username">Username</Label>
                <Input
                  id="login-username"
                  placeholder="Enter your username"
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="login-password">Password</Label>
                <Input
                  id="login-password"
                  type="password"
                  placeholder="Enter your password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                />
              </div>
              <Button
                className="w-full h-12 text-lg font-semibold rounded-full shadow-md hover:shadow-glow transition-all"
                onClick={handleLogin}
                disabled={loading}
              >
                {loading ? "Logging in..." : "Login"}
              </Button>
            </TabsContent>

            <TabsContent value="register" className="space-y-4 animate-fade-in-up">
              <div className="space-y-2">
                <Label htmlFor="register-username">Username</Label>
                <Input
                  id="register-username"
                  placeholder="Choose a username (min 3 chars)"
                  value={registerUsername}
                  onChange={(e) => setRegisterUsername(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="register-password">Password</Label>
                <Input
                  id="register-password"
                  type="password"
                  placeholder="Create password (min 6 chars)"
                  value={registerPassword}
                  onChange={(e) => setRegisterPassword(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirm-password">Confirm Password</Label>
                <Input
                  id="confirm-password"
                  type="password"
                  placeholder="Confirm your password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleRegister()}
                />
              </div>
              <Button
                className="w-full h-12 text-lg font-semibold rounded-full shadow-md hover:shadow-glow transition-all"
                onClick={handleRegister}
                disabled={loading}
              >
                {loading ? "Creating Account..." : "Register"}
              </Button>
            </TabsContent>
          </Tabs>

          <div className="mt-8 pt-6 border-t flex items-center justify-center gap-2">
            <span className="text-sm text-muted-foreground">Theme:</span>
            <Button
              variant="outline"
              size="sm"
              onClick={toggleTheme}
              className="rounded-full transition-all hover-lift"
            >
              {theme === "light" ? (
                <>
                  <Moon className="w-4 h-4 mr-2" />
                  Dark
                </>
              ) : (
                <>
                  <Sun className="w-4 h-4 mr-2" />
                  Light
                </>
              )}
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Auth;