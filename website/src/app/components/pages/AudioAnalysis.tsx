import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion } from "motion/react";
import { ArrowLeft, Play, Pause, Activity, TrendingUp, Zap, WavesIcon, BarChart3 } from "lucide-react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export function AudioAnalysis() {
  const { id } = useParams<{ id: string }>();
  const { userData } = useAppContext();
  const navigate = useNavigate();
  const [isPlaying, setIsPlaying] = useState(false);

  const recording = userData.audioRecordings.find(r => r.id === id);

  useEffect(() => {
    if (!userData.optedIn || !recording) {
      navigate("/");
    }
  }, [userData.optedIn, recording, navigate]);

  if (!recording) return null;

  // Generate time series data for animations
  const timeSeriesData = Array.from({ length: 50 }, (_, i) => ({
    time: i,
    pitch: recording.pitch + (Math.sin(i / 5) * 5 + Math.random() * 3 - 1.5),
    shimmer: recording.shimmer + (Math.cos(i / 4) * 0.3 + Math.random() * 0.2 - 0.1),
    jitter: recording.jitter + (Math.sin(i / 6) * 0.2 + Math.random() * 0.15 - 0.075),
  }));

  // Generate frequency distribution (mock data)
  const frequencyData = Array.from({ length: 20 }, (_, i) => ({
    freq: 100 + i * 50,
    amplitude: Math.max(0, 100 - Math.abs(i - 10) * 8 + Math.random() * 20),
  }));

  // Heatmap data (7 days x 24 hours)
  const heatmapData = Array.from({ length: 7 }, (_, day) =>
    Array.from({ length: 24 }, (_, hour) => ({
      day,
      hour,
      value: Math.random() * 100,
    }))
  ).flat();

  const getDayLabel = (day: number) => ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][day];

  // Get comparison data
  const recentRecording = userData.audioRecordings.filter(r => r.id !== recording.id).sort((a, b) => b.timestamp - a.timestamp)[0];

  const comparisonMetrics = recentRecording ? [
    {
      label: 'Pitch',
      current: recording.pitch,
      previous: recentRecording.pitch,
      unit: 'Hz',
      color: '#818cf8',
      icon: Activity,
    },
    {
      label: 'Shimmer',
      current: recording.shimmer,
      previous: recentRecording.shimmer,
      unit: '%',
      color: '#34d399',
      icon: Zap,
    },
    {
      label: 'Jitter',
      current: recording.jitter,
      previous: recentRecording.jitter,
      unit: '%',
      color: '#fbbf24',
      icon: TrendingUp,
    },
  ] : [];

  return (
    <div className="flex flex-col h-full bg-neutral-950 text-neutral-50 relative z-10 overflow-hidden">

      {/* Header */}
      <header className="px-4 sm:px-6 py-4 flex items-center justify-between border-b border-neutral-900 z-20 bg-neutral-950/80 backdrop-blur-md sticky top-0">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="flex items-center gap-2 text-neutral-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="text-sm font-medium">Back</span>
        </button>
        <h1 className="text-lg sm:text-xl font-bold tracking-tight">Analysis</h1>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="w-10 h-10 bg-indigo-500 hover:bg-indigo-600 rounded-full flex items-center justify-center transition-colors"
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950">
        <div className="max-w-7xl mx-auto space-y-8">

          {/* Hero Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-2 sm:grid-cols-4 gap-4"
          >
            <div className="bg-gradient-to-br from-indigo-500/20 to-indigo-600/5 border border-indigo-500/30 rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <Activity className="w-5 h-5 text-indigo-400" />
                <span className="text-sm text-neutral-400">Pitch</span>
              </div>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", delay: 0.1 }}
                className="text-3xl font-black text-indigo-400"
              >
                {recording.pitch}
              </motion.div>
              <div className="text-xs text-neutral-500 mt-1">Hz</div>
            </div>

            <div className="bg-gradient-to-br from-emerald-500/20 to-emerald-600/5 border border-emerald-500/30 rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="w-5 h-5 text-emerald-400" />
                <span className="text-sm text-neutral-400">Shimmer</span>
              </div>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", delay: 0.2 }}
                className="text-3xl font-black text-emerald-400"
              >
                {recording.shimmer}
              </motion.div>
              <div className="text-xs text-neutral-500 mt-1">%</div>
            </div>

            <div className="bg-gradient-to-br from-amber-500/20 to-amber-600/5 border border-amber-500/30 rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-5 h-5 text-amber-400" />
                <span className="text-sm text-neutral-400">Jitter</span>
              </div>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", delay: 0.3 }}
                className="text-3xl font-black text-amber-400"
              >
                {recording.jitter}
              </motion.div>
              <div className="text-xs text-neutral-500 mt-1">%</div>
            </div>

            <div className="bg-gradient-to-br from-cyan-500/20 to-cyan-600/5 border border-cyan-500/30 rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <WavesIcon className="w-5 h-5 text-cyan-400" />
                <span className="text-sm text-neutral-400">Duration</span>
              </div>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", delay: 0.4 }}
                className="text-3xl font-black text-cyan-400"
              >
                {Math.floor(recording.duration / 60)}:{(recording.duration % 60).toString().padStart(2, '0')}
              </motion.div>
              <div className="text-xs text-neutral-500 mt-1">min:sec</div>
            </div>
          </motion.div>

          {/* Time Series Analysis */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-neutral-900/50 backdrop-blur-sm border border-neutral-800 rounded-3xl p-6"
          >
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-indigo-400" />
              Temporal Dynamics
            </h2>

            <div className="space-y-8">
              {/* Pitch over time */}
              <div>
                <h3 className="text-sm text-neutral-400 mb-3">Pitch Variation (Hz)</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeSeriesData}>
                      <defs>
                        <linearGradient id="pitchGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                      <XAxis dataKey="time" stroke="#737373" tick={{ fontSize: 12 }} />
                      <YAxis stroke="#737373" tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#171717', border: 'none', borderRadius: '8px', color: '#fff' }}
                      />
                      <Area type="monotone" dataKey="pitch" stroke="#818cf8" strokeWidth={2} fill="url(#pitchGradient)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Shimmer & Jitter combined */}
              <div>
                <h3 className="text-sm text-neutral-400 mb-3">Amplitude Perturbation (%)</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timeSeriesData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                      <XAxis dataKey="time" stroke="#737373" tick={{ fontSize: 12 }} />
                      <YAxis stroke="#737373" tick={{ fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#171717', border: 'none', borderRadius: '8px', color: '#fff' }}
                      />
                      <Line type="monotone" dataKey="shimmer" stroke="#34d399" strokeWidth={2} dot={false} name="Shimmer" />
                      <Line type="monotone" dataKey="jitter" stroke="#fbbf24" strokeWidth={2} dot={false} name="Jitter" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Frequency Spectrum */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-neutral-900/50 backdrop-blur-sm border border-neutral-800 rounded-3xl p-6"
          >
            <h2 className="text-xl font-bold mb-6">Frequency Spectrum</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={frequencyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                  <XAxis dataKey="freq" stroke="#737373" tick={{ fontSize: 12 }} label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#737373" tick={{ fontSize: 12 }} label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#171717', border: 'none', borderRadius: '8px', color: '#fff' }}
                  />
                  <Bar dataKey="amplitude" fill="#818cf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Heatmap - Voice Activity Pattern */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-neutral-900/50 backdrop-blur-sm border border-neutral-800 rounded-3xl p-6"
          >
            <h2 className="text-xl font-bold mb-6">Voice Activity Heatmap (Weekly Pattern)</h2>
            <div className="overflow-x-auto">
              <div className="min-w-[600px]">
                {/* Hour labels */}
                <div className="flex mb-2">
                  <div className="w-12"></div>
                  {Array.from({ length: 24 }, (_, i) => (
                    <div key={i} className="flex-1 text-center text-xs text-neutral-500">
                      {i % 4 === 0 ? i : ''}
                    </div>
                  ))}
                </div>

                {/* Heatmap grid */}
                {Array.from({ length: 7 }, (_, day) => (
                  <div key={day} className="flex mb-1">
                    <div className="w-12 text-xs text-neutral-500 flex items-center">
                      {getDayLabel(day)}
                    </div>
                    {Array.from({ length: 24 }, (_, hour) => {
                      const dataPoint = heatmapData.find(d => d.day === day && d.hour === hour);
                      const intensity = dataPoint ? dataPoint.value / 100 : 0;
                      const color = intensity > 0.7 ? 'bg-indigo-500' :
                                   intensity > 0.5 ? 'bg-indigo-600' :
                                   intensity > 0.3 ? 'bg-indigo-700' :
                                   intensity > 0.1 ? 'bg-indigo-800' :
                                   'bg-neutral-800';

                      return (
                        <motion.div
                          key={hour}
                          initial={{ opacity: 0, scale: 0 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: (day * 24 + hour) * 0.002 }}
                          className={`flex-1 aspect-square ${color} rounded-sm mx-0.5 cursor-pointer hover:ring-2 hover:ring-indigo-400 transition-all`}
                          title={`${getDayLabel(day)} ${hour}:00 - Activity: ${Math.round(intensity * 100)}%`}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Comparison to Previous Recording */}
          {recentRecording && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-neutral-900/50 backdrop-blur-sm border border-neutral-800 rounded-3xl p-6"
            >
              <h2 className="text-xl font-bold mb-6">Comparison to Most Recent</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {comparisonMetrics.map((metric, idx) => {
                  const diff = metric.current - metric.previous;
                  const percentChange = ((diff / metric.previous) * 100).toFixed(1);
                  const isIncrease = diff > 0;
                  const Icon = metric.icon;

                  return (
                    <motion.div
                      key={metric.label}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + idx * 0.1 }}
                      className="bg-neutral-800/30 rounded-2xl p-5"
                    >
                      <div className="flex items-center gap-2 mb-4">
                        <Icon className="w-5 h-5" style={{ color: metric.color }} />
                        <span className="text-sm text-neutral-400">{metric.label}</span>
                      </div>

                      <div className="space-y-3">
                        <div>
                          <div className="text-xs text-neutral-500 mb-1">Current</div>
                          <div className="text-2xl font-bold" style={{ color: metric.color }}>
                            {metric.current}{metric.unit}
                          </div>
                        </div>

                        <div>
                          <div className="text-xs text-neutral-500 mb-1">Previous</div>
                          <div className="text-lg text-neutral-400">
                            {metric.previous}{metric.unit}
                          </div>
                        </div>

                        <div className={`text-sm font-semibold ${isIncrease ? 'text-rose-400' : 'text-emerald-400'}`}>
                          {isIncrease ? '↑' : '↓'} {Math.abs(parseFloat(percentChange))}% change
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

        </div>
      </main>
    </div>
  );
}
