// Initialize default admin user if not exists
export const initializeApp = () => {
  const users = JSON.parse(localStorage.getItem("plantdoctor_users") || "[]");
  
  // Check if admin exists
  const adminExists = users.some((u: any) => u.role === "admin");
  
  if (!adminExists) {
    // Hash password "admin123" with SHA-256
    const adminPassword = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"; // SHA-256 hash of "admin123"
    
    const adminUser = {
      id: "admin_" + Date.now(),
      username: "admin",
      password: adminPassword,
      role: "admin",
      createdAt: new Date().toISOString(),
    };
    
    users.push(adminUser);
    localStorage.setItem("plantdoctor_users", JSON.stringify(users));
    console.log("✅ Default admin user created (username: admin, password: admin123)");
  }
};
