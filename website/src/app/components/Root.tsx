import { Outlet } from "react-router";
import { AppProvider } from "../AppContext";

export function Root() {
  return (
    <AppProvider>
      <div className="min-h-screen bg-[#faf8ff] text-slate-900 font-sans selection:bg-violet-300/30 flex justify-center">
        <div className="w-full max-w-7xl bg-white/60 min-h-screen relative lg:shadow-xl lg:shadow-violet-200/20 overflow-hidden flex flex-col border-x border-violet-100/50">
          <Outlet />
        </div>
      </div>
    </AppProvider>
  );
}
