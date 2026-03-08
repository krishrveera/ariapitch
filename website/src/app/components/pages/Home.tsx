import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion, AnimatePresence } from "motion/react";
import { Mic, Activity, Info, X, AlertTriangle, Flame, Music, HelpCircle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { VoiceTermsGlossary } from "../VoiceTermsGlossary";

export function Home() {
  const { userData, setUserData } = useAppContext();
  const navigate = useNavigate();
  const [showPrompt, setShowPrompt] = useState(!userData.hasRecordedToday);
  const [showHealthPopup, setShowHealthPopup] = useState(false);
  const [glossaryOpen, setGlossaryOpen] = useState(false);

  useEffect(() => {
    if (!userData.onboardingComplete) {
      navigate("/onboarding");
    }
    if (userData.showHealthPopup) {
      setShowHealthPopup(true);
    }
  }, [userData.onboardingComplete, userData.showHealthPopup, navigate]);

  const handleRecord = () => {
    navigate("/record");
  };

  const handleBypass = () => {
    setShowPrompt(false);
  };

  const closeHealthPopup = () => {
    setShowHealthPopup(false);
    setUserData(prev => ({ ...prev, showHealthPopup: false }));
  };

  if (!userData.onboardingComplete) return null;

  const latestEntry = userData.history.length > 0 ? userData.history[userData.history.length - 1] : null;

  return (
    <div className="flex flex-col h-full bg-[#faf8ff] text-slate-900 relative z-10">

      {/* Header */}
      <header className="px-4 sm:px-6 py-4 flex items-center justify-between border-b border-violet-100 z-20 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <Activity className="w-6 h-6 text-violet-600" />
          <h1 className="text-lg sm:text-xl font-bold tracking-tight">AriaPitch</h1>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          {userData.optedIn && (
            <button
              onClick={() => navigate("/audio-library")}
              className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full bg-violet-100 text-violet-700 hover:bg-violet-200 transition-colors"
              title="Audio Recordings Library (Bridge2AI)"
            >
              <Music className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="text-xs sm:text-sm font-semibold hidden sm:inline">Library</span>
            </button>
          )}
          <button
            onClick={() => navigate("/record")}
            className="w-9 h-9 rounded-full bg-violet-600 hover:bg-violet-700 text-white flex items-center justify-center transition-colors shadow-md shadow-violet-200/40"
            title="Record"
          >
            <Mic className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full bg-amber-50 text-amber-600 border border-amber-100">
            <Flame className="w-3 h-3 sm:w-4 sm:h-4" />
            <span className="text-xs sm:text-sm font-semibold">{userData.streak ?? 0}</span>
          </div>
          <button
            onClick={() => navigate("/onboarding")}
            className={`text-xs px-2 sm:px-3 py-1.5 rounded-full font-medium transition-colors ${userData.optedIn
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

            {/* Charts - now first */}
            <section className="space-y-4 lg:pb-8">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Biomarkers</h2>

              <div className="bg-white border border-violet-100 rounded-2xl p-4 sm:p-6 space-y-4 shadow-sm">
                <div>
                  <div className="flex justify-between items-end flex-wrap gap-2">
                    <h3 className="text-xs sm:text-sm text-slate-500">Fundamental Frequency (Pitch)</h3>
                    <span className="text-base sm:text-lg font-semibold text-violet-600">{latestEntry ? `${latestEntry.pitch} Hz` : '—'}</span>
                  </div>
                  <div className="h-32 sm:h-40 mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={userData.history} margin={{ top: 5, left: 10, right: 15, bottom: 5 }}>
                        <XAxis dataKey="date" stroke="#a1a1aa" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} tickFormatter={(v: string) => v.split(',')[0]?.slice(0, 3) || v} />
                        <YAxis domain={['dataMin - 10', 'dataMax + 10']} stroke="#a1a1aa" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={45} tickFormatter={(v: number) => `${v}`} />
                        <Line type="monotone" dataKey="pitch" stroke="#7c3aed" strokeWidth={2} dot={{ r: 3, fill: '#7c3aed' }} />
                        <ReferenceLine y={200} stroke="#d4d4d8" strokeDasharray="6 4" strokeWidth={1.5} label={{ value: "Healthy", position: "right", fill: "#a1a1aa", fontSize: 10 }} />
                        <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ede9fe', borderRadius: '12px', color: '#1e1b4b', boxShadow: '0 4px 12px rgba(124,58,237,0.08)' }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="h-px bg-violet-100 my-4" />

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-xs text-slate-500 mb-1">Shimmer</h3>
                    <div className="text-base sm:text-lg font-semibold text-emerald-600">{latestEntry ? `${latestEntry.shimmer} dB` : '—'}</div>
                    <div className="h-16 sm:h-20 mt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={userData.history} margin={{ top: 5, left: 5, right: 5, bottom: 0 }}>
                          <YAxis domain={['dataMin - 0.1', 'dataMax + 0.1']} stroke="#a1a1aa" tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={32} tickFormatter={(v: number) => `${v}`} />
                          <Line type="monotone" dataKey="shimmer" stroke="#10b981" strokeWidth={2} dot={false} />
                          <ReferenceLine y={0.35} stroke="#d4d4d8" strokeDasharray="4 3" strokeWidth={1} label={{ value: "Healthy", position: "right", fill: "#a1a1aa", fontSize: 9 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-xs text-slate-500 mb-1">Harmonic-to-Noise Ratio</h3>
                    <div className="text-base sm:text-lg font-semibold text-purple-600">{latestEntry?.harmonicRatio ? `${latestEntry.harmonicRatio} dB` : '—'}</div>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="flex-1 h-2 bg-purple-100 rounded-full overflow-hidden">
                        <div className="h-full bg-purple-500 rounded-full" style={{ width: `${latestEntry?.harmonicRatio ? Math.min((latestEntry.harmonicRatio / 25) * 100, 100) : 0}%` }} />
                      </div>
                      <span className="text-[10px] text-slate-400">{latestEntry?.harmonicRatio && latestEntry.harmonicRatio > 13 ? 'Good' : 'Low'}</span>
                    </div>
                  </div>
                </div>

                <div className="h-px bg-violet-100 my-4" />

                {/* Learn more button */}
                <button onClick={() => setGlossaryOpen(true)} className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-violet-600 bg-violet-50 hover:bg-violet-100 rounded-full transition-colors">
                  <HelpCircle className="w-3.5 h-3.5" /> What do these biomarkers mean?
                </button>
              </div>
            </section>

            {/* Feed of Messages - newest first */}
            <section className="space-y-4">
              <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Voice Insights</h2>
              <div className="flex flex-col gap-3 max-h-[420px] lg:max-h-[600px] overflow-y-auto pr-1 -mr-1">
                {[...userData.history].reverse().map((entry, i) => {
                  const colors = [
                    { bg: "bg-violet-50", border: "border-violet-200", text: "text-violet-700", meta: "text-violet-400" },
                    { bg: "bg-sky-50", border: "border-sky-200", text: "text-sky-700", meta: "text-sky-400" },
                    { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-700", meta: "text-amber-400" },
                    { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700", meta: "text-emerald-400" },
                    { bg: "bg-pink-50", border: "border-pink-200", text: "text-pink-700", meta: "text-pink-400" },
                    { bg: "bg-teal-50", border: "border-teal-200", text: "text-teal-700", meta: "text-teal-400" },
                  ];
                  const c = entry.isAnomaly
                    ? { bg: "bg-rose-50", border: "border-rose-200", text: "text-rose-700", meta: "text-rose-400" }
                    : colors[i % colors.length];

                  return (
                    <div
                      key={i}
                      className={`self-start max-w-full lg:max-w-[85%] px-3 sm:px-4 py-3 rounded-2xl rounded-tl-md ${c.bg} border ${c.border} shadow-sm`}
                    >
                      <div className="flex justify-between items-start mb-1.5 gap-2">
                        <span className={`text-xs font-medium ${c.meta}`}>{entry.date}</span>
                        {entry.isAnomaly && <AlertTriangle className="w-3.5 h-3.5 text-rose-500 shrink-0" />}
                      </div>
                      <p className={`text-sm leading-relaxed ${c.text}`}>
                        {entry.message}
                      </p>
                      <div className={`mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs ${c.meta}`}>
                        <span>Pitch: {entry.pitch}Hz</span>
                        <span>Shimmer: {entry.shimmer} dB</span>
                        <span>HNR: {entry.harmonicRatio} dB</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

          </div>


        </div>

      </main>

      {/* Daily Voice Prompt Overlay */}
      <AnimatePresence>
        {showPrompt && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="absolute inset-0 z-50 flex items-center justify-center p-6 bg-[#faf8ff]/80 backdrop-blur-md"
          >
            <div className="bg-white border border-violet-200 rounded-3xl p-8 w-full max-w-sm text-center shadow-xl shadow-violet-200/30 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-violet-500 via-purple-400 to-violet-500"></div>

              <div className="w-20 h-20 bg-violet-100 rounded-full flex items-center justify-center mx-auto mb-6 relative">
                <div className="absolute inset-0 bg-violet-300 rounded-full animate-ping opacity-20"></div>
                <Mic className="w-8 h-8 text-violet-600" />
              </div>

              <h2 className="text-2xl font-bold mb-2">Time to Record!</h2>
              <p className="text-slate-500 mb-8 text-sm">Please find a quiet place before you start recording.</p>

              <div className="space-y-3">
                <button
                  onClick={handleRecord}
                  className="w-full py-4 bg-violet-600 hover:bg-violet-700 text-white rounded-xl font-semibold transition-colors flex items-center justify-center gap-2 shadow-lg shadow-violet-300/30"
                >
                  <Mic className="w-5 h-5" /> Start Recording
                </button>
                <button
                  onClick={handleBypass}
                  className="w-full py-3 bg-violet-50 hover:bg-violet-100 text-slate-600 rounded-xl font-medium transition-colors text-sm"
                >
                  Not in a Quiet Place
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Health Recommendation Popup */}
      <AnimatePresence>
        {showHealthPopup && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-[60] flex items-center justify-center p-6 bg-slate-900/20 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="bg-white border border-rose-200 rounded-3xl p-6 w-full max-w-sm shadow-xl"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="w-12 h-12 bg-rose-50 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-rose-500" />
                </div>
                <button onClick={closeHealthPopup} className="p-2 text-slate-400 hover:text-slate-700 transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <h3 className="text-xl font-bold text-slate-900 mb-2">Anomaly Detected</h3>
              <p className="text-slate-600 text-sm leading-relaxed mb-6">
                Your vocal jitter has increased by 40% compared to your baseline. This could indicate early signs of vocal cord inflammation or fatigue. We recommend resting your voice and consulting an ENT if symptoms persist.
              </p>

              <div className="space-y-3">
                <button onClick={closeHealthPopup} className="w-full py-3 bg-rose-500 hover:bg-rose-600 text-white rounded-xl font-semibold transition-colors">
                  Acknowledge
                </button>
                <button className="w-full py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-medium transition-colors text-sm flex items-center justify-center gap-2">
                  <Info className="w-4 h-4" /> Find a Specialist
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Terms Glossary */}
      <VoiceTermsGlossary open={glossaryOpen} onOpenChange={setGlossaryOpen} />
    </div>
  );
}
