const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Helper to get auth token
const getAuthToken = (): string | null => {
  const user = localStorage.getItem("plantdoctor_current_user");
  if (user) {
    const parsed = JSON.parse(user);
    return parsed.access_token || null;
  }
  return null;
};

// Helper for API requests
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getAuthToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Auth API
export const authAPI = {
  async login(username: string, password: string) {
    return apiRequest<{ access_token: string; token_type: string; user: any }>(
      "/auth/login",
      {
        method: "POST",
        body: JSON.stringify({ username, password }),
      }
    );
  },

  async register(username: string, password: string, confirm_password: string) {
    return apiRequest<{ access_token: string; token_type: string; user: any }>(
      "/auth/register",
      {
        method: "POST",
        body: JSON.stringify({ username, password, confirm_password }),
      }
    );
  },
};

// Users API
export const usersAPI = {
  async getCurrentUser() {
    return apiRequest<any>("/users/me");
  },

  async getDashboard() {
    return apiRequest<{
      user: any;
      recent_analyses: any[];
      badges: any[];
      ranking: any;
    }>("/users/dashboard");
  },

  async getUserProfile(username: string) {
    return apiRequest<any>(`/users/profile/${username}`);
  },
};

// History API
export const historyAPI = {
  async addAnalysis(data: {
    image_url: string;
    user_context?: string;
    predictions: Array<{ disease_name: string; confidence: number; model: string }>;
    best_model: string;
    best_disease: string;
    best_confidence: number;
    urgency_level: "low" | "medium" | "high";
  }) {
    return apiRequest<any>("/history/analysis", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  async getHistory(limit?: number) {
    const query = limit ? `?limit=${limit}` : "";
    return apiRequest<any[]>(`/history/${query}`);
  },

  async getStats() {
    return apiRequest<{
      total_analyses: number;
      plant_types: number;
      diseases_found: number;
      healthy_plants: number;
      average_accuracy: number;
      model_performance: any;
      eco_metrics: any;
    }>("/history/stats");
  },
};

// Leaderboard API
export const leaderboardAPI = {
  async getLeaderboard(limit: number = 10, sortBy: string = "points") {
    return apiRequest<{
      top_users: any[];
      current_user: any;
      total_users: number;
    }>(`/leaderboard/?limit=${limit}&sort_by=${sortBy}`);
  },

  async getGlobalStats() {
    return apiRequest<{
      total_users: number;
      total_analyses: number;
      total_diseases_detected: number;
      average_system_accuracy: number;
      top_disease: string;
      most_active_user: string;
    }>("/leaderboard/stats");
  },

  async getTopPerformers() {
    return apiRequest<{
      top_points: any[];
      top_plants_saved: any[];
      top_accuracy: any[];
      top_badges: any[];
    }>("/leaderboard/top-performers");
  },
};

// Badges API
export const badgesAPI = {
  async getBadges() {
    return apiRequest<{
      earned_badges: any[];
      available_badges: any[];
      total_badges: number;
      completion_percentage: number;
    }>("/badges/");
  },

  async getBadgeProgress() {
    return apiRequest<{ badges: any[] }>("/badges/progress");
  },

  async getBadgeLeaderboard() {
    return apiRequest<{ leaderboard: any[] }>("/badges/leaderboard");
  },
};

// Admin API
export const adminAPI = {
  async getDashboard() {
    return apiRequest<any>("/admin/dashboard");
  },

  async getUsers(search?: string, sortBy: string = "points", order: string = "desc") {
    let query = `?sort_by=${sortBy}&order=${order}`;
    if (search) query += `&search=${search}`;
    return apiRequest<{
      total_users: number;
      active_users: number;
      total_badges: number;
      average_scans: number;
      users: any[];
    }>(`/admin/users${query}`);
  },

  async getUserDetails(username: string) {
    return apiRequest<{
      user: any;
      history_count: number;
      badges: any[];
      analytics: any;
    }>(`/admin/users/${username}`);
  },

  async deleteUser(username: string) {
    return apiRequest<{ message: string }>(`/admin/users/${username}`, {
      method: "DELETE",
    });
  },

  async getStatistics() {
    return apiRequest<any>("/admin/statistics");
  },

  async getRecentActivity(limit: number = 10) {
    return apiRequest<{ activity: any[] }>(`/admin/recent-activity?limit=${limit}`);
  },
};

export default {
  auth: authAPI,
  users: usersAPI,
  history: historyAPI,
  leaderboard: leaderboardAPI,
  badges: badgesAPI,
  admin: adminAPI,
};