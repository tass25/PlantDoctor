import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is logged in
    const currentUser = localStorage.getItem("plantdoctor_current_user");
    if (currentUser) {
      const user = JSON.parse(currentUser);
      if (user.role === "admin") {
        navigate("/admin");
      } else {
        navigate("/dashboard");
      }
    } else {
      navigate("/auth");
    }
  }, [navigate]);

  return null;
};

export default Index;
