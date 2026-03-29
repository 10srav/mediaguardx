import { create } from 'zustand';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
}

interface UIState {
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;
  toasts: Toast[];
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebarCollapse: () => void;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
}

// Track toast timer IDs for cleanup
const toastTimers = new Map<string, ReturnType<typeof setTimeout>>();

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: false,
  sidebarCollapsed: false,
  toasts: [],
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebarCollapse: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  addToast: (toast) =>
    set((s) => {
      const id = Math.random().toString(36).slice(2);
      const timer = setTimeout(() => {
        toastTimers.delete(id);
        set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
      }, 5000);
      toastTimers.set(id, timer);
      return { toasts: [...s.toasts, { ...toast, id }] };
    }),
  removeToast: (id) => {
    // Clear the auto-dismiss timer when manually removed
    const timer = toastTimers.get(id);
    if (timer) {
      clearTimeout(timer);
      toastTimers.delete(id);
    }
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  },
  clearToasts: () => {
    // Clear all pending timers
    for (const timer of toastTimers.values()) {
      clearTimeout(timer);
    }
    toastTimers.clear();
    set({ toasts: [] });
  },
}));
