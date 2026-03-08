import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion, AnimatePresence } from "motion/react";
import { Mic, Activity, Info, X, AlertTriangle, Flame, Music } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { RecordedWaveform } from "../ui/RecordedWaveform";
import { Spectrogram } from "../ui/Spectrogram";
import { VoiceTermsGlossary } from "../VoiceTermsGlossary";

// ── Status helpers ────────────────────────────────────────────

type Status = "excellent" | "good" | "normal" | "warning";

function getEntryStatus(entry: { pitch: number; shimmer: number; jitter: number; isAnomaly: boolean }): Status {
  if (entry.isAnomaly) return "warning";
  if (entry.jitter <= 1.0 && entry.shimmer <= 3.2) return "excellent";
  if (entry.jitter <= 1.3 && entry.shimmer <= 3.6) return "good";
  return "normal";
}

const STATUS_CONFIG: Record<Status, { label: string; accent: string; border: string; bg: string; badge: string; text: string }> = {
  excellent: { label: "Excellent", accent: "bg-emerald-500", border: "border-emerald-400/40", bg: "bg-emerald-50/80",   badge: "bg-emerald-100 text-emerald-700", text: "text-emerald-900" },
  good:      { label: "Good",      accent: "bg-sky-500",     border: "border-sky-400/40",     bg: "bg-sky-50/80",       badge: "bg-sky-100 text-sky-700",         text: "text-sky-900" },
  normal:    { label: "Normal",    accent: "bg-amber-500",   border: "border-amber-400/40",   bg: "bg-amber-50/80",     badge: "bg-amber-100 text-amber-700",     text: "text-amber-900" },
  warning:   { label: "Warning",   accent: "bg-rose-500",    border: "border-rose-400/40",    bg: "bg-rose-50/80",      badge: "bg-rose-100 text-rose-700",       text: "text-rose-900" },
};

// ── Custom tooltip ────────────────────────────────────────────

function FrostedTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl px-4 py-3 border border-purple-200/60 shadow-xl" style={{ background: "rgba(255,255,255,0.85)", backdropFilter: "blur(16px)", fontFamily: "var(--font-body)" }}>
      <p className="text-xs text-purple-400 mb-1" style={{ fontFamily: "var(--font-body)" }}>{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-sm font-semibold" style={{ color: p.stroke, fontFamily: "var(--font-mono)" }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </p>
      ))}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────

export function Home() {
  const { userData, setUserData } = useAppContext();
  const navigate = useNavigate();
  const [showPrompt, setShowPrompt] = useState(!userData.hasRecordedToday);
  const [showHealthPopup, setShowHealthPopup] = useState(false);

  useEffect(() => {
    if (!userData.onboardingComplete) {
      navigate("/onboarding");
    }
    if (userData.showHealthPopup) {
      setShowHealthPopup(true);
    }
  }, [userData.onboardingComplete, userData.showHealthPopup, navigate]);

  const handleRecord = () => navigate("/record");
  const handleBypass = () => setShowPrompt(false);

  const closeHealthPopup = () => {
    setShowHealthPopup(false);
    setUserData(prev => ({ ...prev, showHealthPopup: false }));
  };

  if (!userData.onboardingComplete) return null;

  const latestEntry = userData.history.length > 0 ? userData.history[userData.history.length - 1] : null;
  const latestPitch = latestEntry?.pitch ?? 212;
  const latestShimmer = latestEntry?.shimmer ?? 3.4;
  const latestJitter = latestEntry?.jitter ?? 1.2;

  return (
    <div className="flex flex-col h-full text-purple-950 relative z-10" style={{ fontFamily: "var(--font-body)" }}>

      {/* Header */}
      <header className="px-4 sm:px-6 py-4 flex items-center justify-between z-20 border-b border-purple-200/40" style={{ background: "rgba(255,255,255,0.55)", backdropFilter: "blur(16px)" }}>
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <Activity className="w-6 h-6 text-purple-600 relative z-10" />
            <div className="absolute inset-0 w-6 h-6 rounded-full blur-md bg-purple-400/50" />
          </div>
          <h1 className="text-lg sm:text-xl font-bold tracking-tight text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>ArioPitch</h1>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          {userData.optedIn && (
            <button
              onClick={() => navigate("/audio-library")}
              className="flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-full transition-colors border border-purple-300/50 bg-white/60 text-purple-600 hover:bg-purple-50"
              title="Audio Recordings Library (Bridge2AI)"
            >
              <Music className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="text-xs sm:text-sm font-semibold hidden sm:inline">Library</span>
            </button>
          )}
          <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full border border-amber-300/50 bg-amber-50/80 text-amber-600">
            <Flame className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span className="text-xs sm:text-sm font-bold tabular-nums" style={{ fontFamily: "var(--font-mono)" }}>{userData.streak ?? 0}</span>
          </div>
          {!userData.hasRecordedToday && (
            <button
              onClick={handleRecord}
              className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-rose-500 hover:bg-rose-600 text-white flex items-center justify-center shadow-md shadow-rose-500/30 transition-colors"
              title="Record your daily voice"
            >
              <Mic className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>
          )}
          <button
            onClick={() => navigate("/onboarding")}
            className={`text-xs px-2 sm:px-3 py-1.5 rounded-full font-medium transition-colors shadow-sm ${userData.optedIn
                ? "bg-emerald-500 text-white hover:bg-emerald-600"
                : "bg-rose-500 text-white hover:bg-rose-600"
              }`}
          >
            <span className="hidden sm:inline">Opt In/Out of Study</span>
            <span className="sm:hidden">Opt {userData.optedIn ? 'In' : 'Out'}</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className={`flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 ${showPrompt ? "blur-sm opacity-50 pointer-events-none" : ""}`}>

        <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8">

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">

            {/* ── Trend Analysis (now primary) with attached mini summary ── */}
            <section className="space-y-4 lg:pb-8">
              <h2 className="text-[11px] font-semibold text-purple-400 uppercase tracking-[0.15em]" style={{ fontFamily: "var(--font-body)" }}>Trend Analysis</h2>

              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15, duration: 0.45 }}
                className="rounded-2xl border border-purple-200/50 bg-white/70 backdrop-blur-md p-4 sm:p-6 space-y-4 shadow-sm hover:translate-y-[-1px] hover:shadow-md transition-all duration-200"
              >
                {/* Compact biomarker summary chips, attached to the trend card */}
                <div className="flex flex-wrap gap-2 mb-2">
                  {[
                    { label: "Pitch", value: latestPitch, unit: "Hz", color: "#7c3aed" },
                    { label: "Shimmer", value: latestShimmer, unit: "%", color: "#059669" },
                    { label: "Jitter", value: latestJitter, unit: "%", color: "#d97706" },
                  ].map(metric => (
                    <div
                      key={metric.label}
                      className="px-3 py-1.5 rounded-full bg-white/80 border border-purple-200/70 flex items-baseline gap-1.5 shadow-xs"
                    >
                      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-purple-400" style={{ fontFamily: "var(--font-body)" }}>
                        {metric.label}
                      </span>
                      <span
                        className="text-sm font-bold tabular-nums"
                        style={{ color: metric.color, fontFamily: "var(--font-mono)" }}
                      >
                        {typeof metric.value === "number" ? metric.value.toFixed(1) : metric.value}
                      </span>
                      <span className="text-[10px] text-purple-300" style={{ fontFamily: "var(--font-mono)" }}>
                        {metric.unit}
                      </span>
                    </div>
                  ))}
                </div>

                <div>
                  <div className="flex justify-between items-end flex-wrap gap-2">
                    <h3 className="text-[10px] sm:text-xs font-semibold uppercase tracking-[0.12em] text-purple-400" style={{ fontFamily: "var(--font-body)" }}>Fundamental Frequency (Pitch)</h3>
                    <span className="text-base sm:text-lg font-bold tabular-nums text-purple-600" style={{ fontFamily: "var(--font-mono)" }}>~{latestPitch} Hz</span>
                  </div>
                  <div className="h-36 sm:h-44 mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={userData.history}>
                        <defs>
                          <linearGradient id="pitchGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <Area type="monotone" dataKey="pitch" stroke="#8b5cf6" strokeWidth={3.2} fill="url(#pitchGradient)" dot={{ r: 3.8, fill: '#8b5cf6', strokeWidth: 0 }} name="Pitch" />
                        <ReferenceLine y={200} stroke="#c4b5fd" strokeDasharray="6 4" strokeWidth={1} label={{ value: "Baseline", position: "right", fill: "#a78bfa", fontSize: 10 }} />
                        <YAxis hide domain={['dataMin - 10', 'dataMax + 10']} />
                        <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#a78bfa' }} axisLine={false} tickLine={false} />
                        <Tooltip content={<FrostedTooltip />} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="h-px bg-purple-200/40 my-4" />

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-purple-400 mb-1" style={{ fontFamily: "var(--font-body)" }}>Shimmer</h3>
                    <div className="text-base sm:text-lg font-bold text-emerald-600 tabular-nums" style={{ fontFamily: "var(--font-mono)" }}>{latestShimmer}%</div>
                    <div className="h-16 sm:h-20 mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={userData.history}>
                          <defs>
                            <linearGradient id="shimmerGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#10b981" stopOpacity={0.25} />
                              <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
                            </linearGradient>
                          </defs>
                          <Area type="monotone" dataKey="shimmer" stroke="#10b981" strokeWidth={2.6} fill="url(#shimmerGradient)" dot={false} name="Shimmer" />
                          <ReferenceLine y={3.0} stroke="#a7f3d0" strokeDasharray="4 3" strokeWidth={1} />
                          <YAxis hide domain={['dataMin - 0.5', 'dataMax + 0.5']} />
                          <Tooltip content={<FrostedTooltip />} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-purple-400 mb-1" style={{ fontFamily: "var(--font-body)" }}>Jitter</h3>
                    <div className="text-base sm:text-lg font-bold text-amber-600 tabular-nums" style={{ fontFamily: "var(--font-mono)" }}>{latestJitter}%</div>
                    <div className="h-16 sm:h-20 mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={userData.history}>
                          <defs>
                            <linearGradient id="jitterGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.25} />
                              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.02} />
                            </linearGradient>
                          </defs>
                          <Area type="monotone" dataKey="jitter" stroke="#f59e0b" strokeWidth={2.6} fill="url(#jitterGradient)" dot={false} name="Jitter" />
                          <ReferenceLine y={1.0} stroke="#fde68a" strokeDasharray="4 3" strokeWidth={1} />
                          <YAxis hide domain={['dataMin - 0.3', 'dataMax + 0.3']} />
                          <Tooltip content={<FrostedTooltip />} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </motion.div>
            </section>

            {/* ── Voice Insights — newest first ── */}
            <section className="space-y-4">
              <h2 className="text-[11px] font-semibold text-purple-400 uppercase tracking-[0.15em]" style={{ fontFamily: "var(--font-body)" }}>Voice Insights</h2>
              <div className="flex flex-col gap-3 max-h-[420px] lg:max-h-[600px] overflow-y-auto pr-1 -mr-1">
                {[...userData.history].reverse().map((entry, i) => {
                  const status = getEntryStatus(entry);
                  const cfg = STATUS_CONFIG[status];
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.04, duration: 0.35 }}
                      className={`w-full flex rounded-2xl rounded-tl-md border backdrop-blur-md shadow-sm ${cfg.border} ${cfg.bg} overflow-hidden hover:translate-y-[-1px] hover:shadow-md transition-all duration-200`}
                    >
                      {/* Left accent bar */}
                      <div className={`w-1.5 shrink-0 ${cfg.accent}`} />
                      <div className="px-4 py-3 flex-1 min-w-0">
                        <div className="flex justify-between items-center mb-1.5 gap-2">
                          <span className="text-[10px] font-medium text-purple-400">{entry.date}</span>
                          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${cfg.badge}`}>{cfg.label}</span>
                        </div>
                        <p className={`text-sm leading-relaxed ${cfg.text}`} style={{ fontFamily: "var(--font-body)" }}>
                          {entry.message}
                        </p>
                        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-purple-400" style={{ fontFamily: "var(--font-mono)" }}>
                          <span>Pitch: {entry.pitch}Hz</span>
                          <span>Shimmer: {entry.shimmer}%</span>
                          <span>Jitter: {entry.jitter}%</span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </section>

          </div>

          {/* ── Latest Recording Waveform & Spectrogram ────── */}
          {userData.lastRecordingData && userData.lastRecordingData.length > 0 && (
            <motion.section
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25, duration: 0.45 }}
              className="space-y-4"
            >
              <h2 className="text-[11px] font-semibold text-purple-400 uppercase tracking-[0.15em]" style={{ fontFamily: "var(--font-body)" }}>Latest Recording</h2>
              <div className="rounded-2xl border border-purple-200/50 bg-white/70 backdrop-blur-md p-4 sm:p-6 space-y-6 shadow-sm">
                <div>
                  <h3 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-purple-400 mb-3" style={{ fontFamily: "var(--font-body)" }}>Waveform Analysis</h3>
                  <RecordedWaveform
                    audioData={userData.lastRecordingData}
                    jitter={latestEntry?.jitter ?? 1.2}
                    shimmer={latestEntry?.shimmer ?? 3.4}
                  />
                </div>
                <div className="h-px bg-purple-200/40" />
                <div>
                  <h3 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-purple-400 mb-3" style={{ fontFamily: "var(--font-body)" }}>Spectrogram</h3>
                  <Spectrogram
                    audioData={userData.lastRecordingData}
                    sampleRate={userData.lastRecordingSampleRate ?? 44100}
                  />
                  <p className="text-xs text-purple-400 mt-2" style={{ fontFamily: "var(--font-body)" }}>
                    Frequency distribution over time — brighter colors indicate stronger signal.
                  </p>
                </div>
              </div>
            </motion.section>
          )}

        </div>

      </main>

      {/* ── Daily Voice Prompt Overlay ──────────────────── */}
      <AnimatePresence>
        {showPrompt && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="absolute inset-0 z-50 flex items-center justify-center p-6" style={{ background: "rgba(139,92,246,0.12)", backdropFilter: "blur(20px)" }}
          >
            <div className="rounded-3xl p-8 w-full max-w-sm text-center shadow-2xl relative overflow-hidden border border-purple-200/50" style={{ background: "rgba(255,255,255,0.88)", backdropFilter: "blur(24px)" }}>
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 via-violet-400 to-fuchsia-400"></div>

              <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6 relative">
                <div className="absolute inset-0 bg-purple-400 rounded-full animate-ping opacity-15"></div>
                <Mic className="w-8 h-8 text-purple-600" />
              </div>

              <h2 className="text-2xl font-bold mb-2 text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Time to Record!</h2>
              <p className="text-purple-500 mb-8 text-sm" style={{ fontFamily: "var(--font-body)" }}>Capture your daily voice biomarker.</p>

              <div className="space-y-3">
                <button
                  onClick={handleRecord}
                  className="w-full py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-semibold transition-colors flex items-center justify-center gap-2 shadow-lg shadow-purple-500/25"
                >
                  <Mic className="w-5 h-5" /> Start Recording
                </button>
                <button
                  onClick={handleBypass}
                  className="w-full py-3 bg-purple-50 hover:bg-purple-100 text-purple-600 rounded-xl font-medium transition-colors text-sm border border-purple-200/60"
                >
                  Not in a Quiet Place
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Health Recommendation Popup ─────────────────── */}
      <AnimatePresence>
        {showHealthPopup && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-[60] flex items-center justify-center p-6" style={{ background: "rgba(139,92,246,0.15)", backdropFilter: "blur(12px)" }}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="rounded-3xl p-6 w-full max-w-sm shadow-2xl border border-rose-300/40" style={{ background: "rgba(255,255,255,0.92)", backdropFilter: "blur(20px)" }}
            >
              <div className="flex justify-between items-start mb-4">
                <div className="w-12 h-12 bg-rose-100 rounded-full flex items-center justify-center border border-rose-200">
                  <AlertTriangle className="w-6 h-6 text-rose-500" />
                </div>
                <button onClick={closeHealthPopup} className="p-2 text-purple-300 hover:text-purple-600 transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <h3 className="text-xl font-bold text-rose-800 mb-2" style={{ fontFamily: "var(--font-brand)" }}>Anomaly Detected</h3>
              <p className="text-purple-700 text-sm leading-relaxed mb-6" style={{ fontFamily: "var(--font-body)" }}>
                Your vocal jitter has increased by 40% compared to your baseline. This could indicate early signs of vocal cord inflammation or fatigue. We recommend resting your voice and consulting an ENT if symptoms persist.
              </p>

              <div className="space-y-3">
                <button onClick={closeHealthPopup} className="w-full py-3 bg-rose-500 hover:bg-rose-600 text-white rounded-xl font-semibold transition-colors shadow-sm">
                  Acknowledge
                </button>
                <button className="w-full py-3 bg-purple-50 hover:bg-purple-100 text-purple-700 rounded-xl font-medium transition-colors text-sm flex items-center justify-center gap-2 border border-purple-200/60">
                  <Info className="w-4 h-4" /> Find a Specialist
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <VoiceTermsGlossary />
    </div>
  );
}
