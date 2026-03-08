import { Outlet } from "react-router";
import { AppProvider } from "../AppContext";

function AnimatedLines() {
  return (
    <svg className="pointer-events-none fixed inset-0 w-full h-full z-0 opacity-[0.18]" viewBox="0 0 1600 900" preserveAspectRatio="none">
      {/* Flowing purple lines — sets of curves at different offsets */}
      {[0, 20, 40, 60, 80].map((offset, groupIdx) => (
        <g key={groupIdx}>
          {[0, 6, 12, 18].map((lineOffset) => (
            <path
              key={lineOffset}
              d={`M-100,${160 + offset + lineOffset + groupIdx * 140} C200,${100 + offset + lineOffset + groupIdx * 140} 400,${260 + offset + lineOffset + groupIdx * 140} 600,${180 + offset + lineOffset + groupIdx * 140} S900,${80 + offset + lineOffset + groupIdx * 140} 1100,${200 + offset + lineOffset + groupIdx * 140} S1400,${280 + offset + lineOffset + groupIdx * 140} 1700,${160 + offset + lineOffset + groupIdx * 140}`}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="1.2"
              strokeOpacity={0.5 - lineOffset * 0.02}
            >
              <animate
                attributeName="d"
                values={`M-100,${160 + offset + lineOffset + groupIdx * 140} C200,${100 + offset + lineOffset + groupIdx * 140} 400,${260 + offset + lineOffset + groupIdx * 140} 600,${180 + offset + lineOffset + groupIdx * 140} S900,${80 + offset + lineOffset + groupIdx * 140} 1100,${200 + offset + lineOffset + groupIdx * 140} S1400,${280 + offset + lineOffset + groupIdx * 140} 1700,${160 + offset + lineOffset + groupIdx * 140};M-100,${180 + offset + lineOffset + groupIdx * 140} C200,${240 + offset + lineOffset + groupIdx * 140} 400,${120 + offset + lineOffset + groupIdx * 140} 600,${210 + offset + lineOffset + groupIdx * 140} S900,${180 + offset + lineOffset + groupIdx * 140} 1100,${140 + offset + lineOffset + groupIdx * 140} S1400,${220 + offset + lineOffset + groupIdx * 140} 1700,${190 + offset + lineOffset + groupIdx * 140};M-100,${160 + offset + lineOffset + groupIdx * 140} C200,${100 + offset + lineOffset + groupIdx * 140} 400,${260 + offset + lineOffset + groupIdx * 140} 600,${180 + offset + lineOffset + groupIdx * 140} S900,${80 + offset + lineOffset + groupIdx * 140} 1100,${200 + offset + lineOffset + groupIdx * 140} S1400,${280 + offset + lineOffset + groupIdx * 140} 1700,${160 + offset + lineOffset + groupIdx * 140}`}
                dur={`${8 + groupIdx * 2 + lineOffset * 0.5}s`}
                repeatCount="indefinite"
              />
            </path>
          ))}
        </g>
      ))}
    </svg>
  );
}

export function Root() {
  return (
    <AppProvider>
      <div
        className="min-h-screen selection:bg-purple-300/40 flex justify-center relative overflow-hidden animate-gradient-bg"
        style={{
          fontFamily: "var(--font-body)",
          background: "linear-gradient(135deg, #f3e8ff 0%, #ede5ff 20%, #e8dff8 40%, #f0e6ff 60%, #ece2ff 80%, #f5edff 100%)",
          backgroundSize: "300% 300%",
        }}
      >
        <AnimatedLines />
        <div className="w-full max-w-7xl min-h-screen relative z-10 overflow-hidden flex flex-col">
          <Outlet />
        </div>
      </div>
    </AppProvider>
  );
}
