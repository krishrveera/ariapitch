import { useNavigate } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion } from "motion/react";
import { ArrowLeft, Play, Calendar, WavesIcon } from "lucide-react";

export function AudioLibrary() {
  const { userData } = useAppContext();
  const navigate = useNavigate();

  if (!userData.optedIn) {
    navigate("/");
    return null;
  }

  const sortedRecordings = [...userData.audioRecordings].sort((a, b) => b.timestamp - a.timestamp);

  return (
    <div className="flex flex-col h-full bg-neutral-950 text-neutral-50 relative z-10">

      {/* Header */}
      <header className="px-4 sm:px-6 py-4 flex items-center justify-between border-b border-neutral-900 z-20 bg-neutral-950">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="flex items-center gap-2 text-neutral-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="text-sm font-medium">Back</span>
        </button>
        <h1 className="text-lg sm:text-xl font-bold tracking-tight">Audio Recordings</h1>
        <div className="w-16" />
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
        <div className="max-w-6xl mx-auto">

          {sortedRecordings.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-24 h-24 bg-indigo-500/10 rounded-full flex items-center justify-center mb-6">
                <WavesIcon className="w-12 h-12 text-indigo-400" />
              </div>
              <h2 className="text-2xl font-bold mb-2">No Recordings Yet</h2>
              <p className="text-neutral-400 max-w-md">
                Complete your first daily voice check to start building your audio library.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {sortedRecordings.map((recording) => (
                <motion.div
                  key={recording.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  whileHover={{ y: -4 }}
                  className={`bg-neutral-900 border rounded-2xl p-5 cursor-pointer transition-all ${
                    recording.isAnomaly
                      ? "border-rose-500/30 hover:border-rose-500/50"
                      : "border-neutral-800 hover:border-indigo-500/50"
                  }`}
                  onClick={() => navigate(`/audio-analysis/${recording.id}`)}
                >
                  {/* Recording Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <div className="w-10 h-10 bg-indigo-500/20 rounded-full flex items-center justify-center">
                        <Play className="w-5 h-5 text-indigo-400" />
                      </div>
                      <div>
                        <div className="text-xs text-neutral-500 flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          {new Date(recording.timestamp).toLocaleDateString()}
                        </div>
                        <div className="text-xs text-neutral-400">
                          {new Date(recording.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                    {recording.isAnomaly && (
                      <div className="w-2 h-2 bg-rose-500 rounded-full animate-pulse" />
                    )}
                  </div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <div className="bg-neutral-800/50 rounded-lg p-2">
                      <div className="text-[10px] text-neutral-500 mb-1">Pitch</div>
                      <div className="text-sm font-bold text-indigo-400">{recording.pitch}Hz</div>
                    </div>
                    <div className="bg-neutral-800/50 rounded-lg p-2">
                      <div className="text-[10px] text-neutral-500 mb-1">Shimmer</div>
                      <div className="text-sm font-bold text-emerald-400">{recording.shimmer}%</div>
                    </div>
                    <div className="bg-neutral-800/50 rounded-lg p-2">
                      <div className="text-[10px] text-neutral-500 mb-1">Jitter</div>
                      <div className="text-sm font-bold text-amber-400">{recording.jitter}%</div>
                    </div>
                  </div>

                  {/* Duration */}
                  <div className="text-xs text-neutral-500">
                    Duration: {Math.floor(recording.duration / 60)}:{(recording.duration % 60).toString().padStart(2, '0')}
                  </div>
                </motion.div>
              ))}
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
