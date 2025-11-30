import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { initializeApp } from "./utils/initializeApp";

// Initialize app with default data
initializeApp();

createRoot(document.getElementById("root")!).render(<App />);
