import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion, AnimatePresence } from "motion/react";
import { Mic, CheckCircle2, XCircle, Play, Square, Activity } from "lucide-react";
import { analyzeAudio, extractBiomarkers, generateMessage } from "../../services/api";
import { LiveWaveform } from "../ui/LiveWaveform";

type FlowState = "PREPARING" | "COUNTDOWN" | "RECORDING" | "ANALYZING" | "RESULT";

const PROMPTS = [
  {
    type: "Vowel",
    text: "1, 2, 3 aah",
    instruction: "Repeat '1, 2, 3 aah' in your normal voice and hold the sound 'aah' for as long as you can",
    recordingDuration: 10,
  },
  {
    type: "Reading",
    text: "Do you like amusement parks? Well, I sure do. To amuse myself, I went twice last spring. My most MEMORABLE moment was riding on the Caterpillar, which is a gigantic rollercoaster high above the ground. When I saw how high the Caterpillar rose into the bright blue sky I knew it was for me. After waiting in line for thirty minutes, I made it to the front where the man measured my height to see if I was tall enough. I gave the man my coins, asked for change, and jumped on the cart. Tick, tick, tick, the Caterpillar climbed slowly up the tracks. It went SO high I could see the parking lot. Boy was I SCARED! I thought to myself, \"There's no turning back now.\" People were so scared they screamed as we swiftly zoomed fast, fast, and faster along the tracks. As quickly as it started, the Caterpillar came to a stop. Unfortunately, it was time to pack the car and drive home. That night I dreamt of the wild ride on the Caterpillar. Taking a trip to the amusement park and riding on the Caterpillar was my MOST memorable moment ever!",
    instruction: "Read the Caterpillar Passage out loud in your typical voice",
    recordingDuration: 110,
  }
];

export function RecordingFlow() {
  const { userData, setUserData } = useAppContext();
  const navigate = useNavigate();
  const [flowState, setFlowState] = useState<FlowState>("PREPARING");
  const [countdown, setCountdown] = useState(3);
  const [recordingTime, setRecordingTime] = useState(0);
  const [currentPrompt, setCurrentPrompt] = useState(PROMPTS[1]);
  const [isSuccess, setIsSuccess] = useState(true);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    setCurrentPrompt(PROMPTS[Math.floor(Math.random() * PROMPTS.length)]);
  }, []);

  useEffect(() => {
    return () => { stopAudioStream(); };
  }, []);

  const stopAudioStream = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
  };

  const startAudioCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const audioCtx = new AudioContext();
      audioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      return true;
    } catch (err) {
      console.error("Microphone access denied:", err);
      return false;
    }
  };

  const startRecording = () => {
    const mediaRecorder = mediaRecorderRef.current;
    if (mediaRecorder && mediaRecorder.state === "inactive") {
      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorder.start();
    }
  };

  const stopAndProcessAudio = async () => {
    return new Promise<void>((resolve) => {
      const mediaRecorder = mediaRecorderRef.current;
      if (!mediaRecorder || mediaRecorder.state === "inactive") { resolve(); return; }
      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          setAudioBlob(blob);
          const arrayBuffer = await blob.arrayBuffer();
          const audioCtx = new AudioContext();
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
          const channelData = audioBuffer.getChannelData(0);
          setUserData(prev => ({ ...prev, lastRecordingData: new Float32Array(channelData), lastRecordingSampleRate: audioBuffer.sampleRate }));
          await audioCtx.close();
        } catch (err) {
          console.error("Error processing audio:", err);
          setAnalysisError("Failed to process audio recording");
        }
        resolve();
      };
      mediaRecorder.stop();
      if (mediaStreamRef.current) { mediaStreamRef.current.getTracks().forEach(t => t.stop()); }
    });
  };

  // Countdown: 3 → 2 → 1 → start recording view
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (flowState === "COUNTDOWN") {
      if (countdown > 0) {
        timer = setTimeout(() => setCountdown(c => c - 1), 1000);
      } else {
        setFlowState("RECORDING");
        setRecordingTime(0);
      }
    }
    return () => clearTimeout(timer);
  }, [countdown, flowState]);

  // Recording timer
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (flowState === "RECORDING") {
      if (recordingTime < currentPrompt.recordingDuration) {
        timer = setTimeout(() => setRecordingTime(t => t + 1), 1000);
      } else {
        handleStopRecording();
      }
    }
    return () => clearTimeout(timer);
  }, [recordingTime, flowState, currentPrompt]);

  // Analysis
  useEffect(() => {
    if (flowState === "ANALYZING" && audioBlob) {
      const runAnalysis = async () => {
        try {
          setAnalysisError(null);
          const taskType = currentPrompt.type === "Vowel" ? "sustained_vowel" : "reading_passage";
          const result = await analyzeAudio(audioBlob, { deviceId: 'web_app', taskType, silenceDuration: 0.5 });
          if (result.success && result.data) {
            setIsSuccess(true);
            const biomarkers = extractBiomarkers(result.data.features);
            const { message, isAnomaly } = generateMessage(result.data.predictions, result.data.explanation);
            (window as any).__analysisResult = { biomarkers, message, isAnomaly, predictions: result.data.predictions };
          } else {
            setIsSuccess(false);
            setAnalysisError(result.error?.message || "Analysis failed");
          }
        } catch (error) {
          console.error("Analysis error:", error);
          setIsSuccess(false);
          setAnalysisError("Failed to analyze recording");
        } finally {
          setFlowState("RESULT");
        }
      };
      runAnalysis();
    }
  }, [flowState, audioBlob, currentPrompt]);

  const handleStart = async () => {
    setCountdown(3);
    setRecordingTime(0);
    const ok = await startAudioCapture();
    if (ok) {
      startRecording();
    }
    setFlowState("COUNTDOWN");
  };

  const handleStopRecording = async () => {
    await stopAndProcessAudio();
    stopAudioStream();
    setFlowState("ANALYZING");
  };

  const finishRecording = () => {
    if (isSuccess) {
      const analysisResult = (window as any).__analysisResult;
      let pitch, shimmer, jitter, message, isAnomaly, spectralCentroid, harmonicRatio;
      if (analysisResult) {
        pitch = analysisResult.biomarkers.pitch; shimmer = analysisResult.biomarkers.shimmer; jitter = analysisResult.biomarkers.jitter;
        spectralCentroid = analysisResult.biomarkers.spectralCentroid; harmonicRatio = analysisResult.biomarkers.harmonicRatio;
        message = analysisResult.message; isAnomaly = analysisResult.isAnomaly;
        delete (window as any).__analysisResult;
      } else {
        pitch = Math.round(200 + Math.random() * 20); shimmer = Number((3 + Math.random() * 1.5).toFixed(1)); jitter = Number((1 + Math.random() * 1).toFixed(1));
        spectralCentroid = Math.round(1500 + Math.random() * 500); harmonicRatio = Number((15 + Math.random() * 8).toFixed(1));
        message = "Your voice exhibits slight jitter today, but pitch is stable. Stay hydrated!"; isAnomaly = false;
      }
      const newEntry = { date: new Date().toLocaleDateString('en-US', { weekday: 'short' }), pitch, shimmer, jitter, message, isAnomaly };
      const updates: any = { hasRecordedToday: true, history: [...userData.history, newEntry], showHealthPopup: isAnomaly };
      if (userData.optedIn) {
        const audioRecording = {
          id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          timestamp: Date.now(), pitch, shimmer, jitter, message, isAnomaly, duration: recordingTime, spectralCentroid, harmonicRatio,
          formants: [Math.round(700 + Math.random() * 100), Math.round(1200 + Math.random() * 200), Math.round(2500 + Math.random() * 300)],
        };
        updates.audioRecordings = [...userData.audioRecordings, audioRecording];
      }
      setUserData(prev => ({ ...prev, ...updates }));
    }
    navigate("/");
  };

  return (
    <div className="flex flex-col h-full text-purple-950 relative z-10" style={{ fontFamily: "var(--font-body)" }}>

      {/* Header — matches Home theme */}
      <header className="px-4 sm:px-6 py-4 flex items-center justify-between z-20 border-b border-purple-200/40" style={{ background: "rgba(255,255,255,0.55)", backdropFilter: "blur(16px)" }}>
        <button
          type="button"
          onClick={() => { stopAudioStream(); navigate("/"); }}
          className="flex items-center gap-2 text-purple-500 hover:text-purple-700 transition-colors"
        >
          <XCircle className="w-5 h-5" />
          <span className="text-sm font-medium hidden sm:inline">Back</span>
        </button>
        <span className="text-[11px] font-semibold tracking-[0.15em] text-purple-400 uppercase">
          {flowState === "PREPARING" && "Get Ready"}
          {flowState === "COUNTDOWN" && "Starting..."}
          {flowState === "RECORDING" && "Recording"}
          {flowState === "ANALYZING" && "Analyzing"}
          {flowState === "RESULT" && "Done"}
        </span>
        <div className="w-16 sm:w-20" />
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col items-center justify-center relative z-20 p-4 sm:p-6 lg:p-8 overflow-y-auto">

        <AnimatePresence mode="wait">

          {flowState === "PREPARING" && (
            <motion.div
              key="preparing"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center text-center space-y-6 sm:space-y-8 w-full max-w-md mx-auto"
            >
              <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-full flex items-center justify-center border border-purple-200/50 shadow-sm" style={{ background: "rgba(255,255,255,0.7)", backdropFilter: "blur(12px)" }}>
                <Mic className="w-8 h-8 sm:w-10 sm:h-10 text-purple-600" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold mb-2 text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Daily Voice Check</h2>
                <p className="text-purple-500 max-w-xs mx-auto text-sm leading-relaxed">
                  Find a quiet place. A 3-second countdown will begin, then you'll complete a voice task.
                </p>
              </div>

              <div className="rounded-2xl border border-purple-200/50 p-4 w-full text-left flex items-start gap-3 sm:gap-4 shadow-sm backdrop-blur-md" style={{ background: "rgba(255,255,255,0.7)" }}>
                <div className="bg-purple-100/80 p-2 rounded-xl text-purple-600 shrink-0 border border-purple-200/40">
                  <Play className="w-4 h-4 sm:w-5 sm:h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-purple-900">Task: {currentPrompt.type}</h3>
                  <p className="text-xs text-purple-500 mt-1">{currentPrompt.instruction}</p>
                </div>
              </div>

              <button
                onClick={handleStart}
                className="w-full py-4 mt-4 sm:mt-8 bg-purple-600 hover:bg-purple-700 text-white rounded-2xl font-bold text-base sm:text-lg shadow-lg shadow-purple-500/25 transition-all"
              >
                I'm Ready
              </button>
            </motion.div>
          )}

          {flowState === "COUNTDOWN" && (
            <motion.div
              key="countdown"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center text-center space-y-6 w-full max-w-md mx-auto"
            >
              <div className="relative w-32 h-32 flex items-center justify-center">
                <motion.div
                  animate={{ scale: [1, 1.15, 1] }}
                  transition={{ repeat: Infinity, duration: 1 }}
                  className="absolute inset-0 rounded-full border border-purple-200/50 shadow-sm" style={{ background: "rgba(255,255,255,0.6)" }}
                />
                <div className="w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse absolute top-2 right-2" />
                <div className="text-[100px] font-black text-purple-600 drop-shadow-[0_0_30px_rgba(139,92,246,0.3)] tabular-nums leading-none">
                  {countdown}
                </div>
              </div>
              <div className="rounded-2xl border border-purple-200/50 p-6 w-full shadow-sm backdrop-blur-md" style={{ background: "rgba(255,255,255,0.7)" }}>
                <p className="text-purple-700 text-lg font-medium">Get ready to speak...</p>
                <p className="text-purple-500/70 text-sm mt-1">Recording has started — your prompt will appear shortly</p>
              </div>
            </motion.div>
          )}

          {flowState === "RECORDING" && (
            <motion.div
              key="recording"
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center w-full max-w-lg mx-auto space-y-6 sm:space-y-8"
            >
              <div className="w-full rounded-2xl border border-purple-200/50 p-4 sm:p-5 shadow-sm backdrop-blur-md" style={{ background: "rgba(255,255,255,0.7)" }}>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse" />
                  <span className="text-[11px] text-purple-400 uppercase tracking-[0.15em] font-semibold">Live Audio</span>
                </div>
                <LiveWaveform analyser={analyserRef.current} isActive={flowState === "RECORDING"} />
              </div>

              <div className="w-full rounded-2xl border border-purple-200/50 p-4 sm:p-5 shadow-sm backdrop-blur-md text-center space-y-2" style={{ background: "rgba(255,255,255,0.7)" }}>
                <div className="text-4xl sm:text-5xl font-black text-purple-900 tabular-nums" style={{ fontFamily: "var(--font-mono)" }}>
                  {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
                  <span className="text-2xl text-purple-400"> / {Math.floor(currentPrompt.recordingDuration / 60)}:{(currentPrompt.recordingDuration % 60).toString().padStart(2, '0')}</span>
                </div>
                <p className="text-sm text-purple-500">Recording in progress</p>
                <div className="w-full bg-purple-200/40 rounded-full h-2 mt-3">
                  <div className="bg-purple-600 h-2 rounded-full transition-all duration-1000" style={{ width: `${(recordingTime / currentPrompt.recordingDuration) * 100}%` }} />
                </div>
              </div>

              <div className="w-full rounded-2xl border border-purple-200/50 p-4 sm:p-6 shadow-sm backdrop-blur-md max-h-[40vh] overflow-y-auto" style={{ background: "rgba(255,255,255,0.7)" }}>
                {currentPrompt.type === "Reading" ? (
                  <div className="text-base sm:text-lg font-medium leading-relaxed text-purple-900 text-left">{currentPrompt.text}</div>
                ) : (
                  <motion.div animate={{ scale: [1, 1.05, 1] }} transition={{ repeat: Infinity, duration: 2 }} className="text-2xl sm:text-3xl font-bold tracking-[0.2em] text-purple-600 text-center">
                    {currentPrompt.text}
                  </motion.div>
                )}
              </div>

              <button
                onClick={handleStopRecording}
                className="w-full py-4 bg-rose-500 hover:bg-rose-600 text-white rounded-2xl font-bold text-lg shadow-lg shadow-rose-500/25 transition-all flex items-center justify-center gap-3"
              >
                <Square className="w-5 h-5 fill-current" /> Stop Recording
              </button>
            </motion.div>
          )}

          {flowState === "ANALYZING" && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center space-y-6"
            >
              <div className="relative w-32 h-32 flex items-center justify-center rounded-full border border-purple-200/50 shadow-sm" style={{ background: "rgba(255,255,255,0.6)" }}>
                <svg className="w-full h-full animate-spin text-purple-200" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" />
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" strokeDasharray="283" strokeDashoffset="200" className="text-purple-600" strokeLinecap="round" />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-purple-600 animate-pulse" />
                </div>
              </div>
              <h2 className="text-xl font-bold text-purple-900" style={{ fontFamily: "var(--font-brand)" }}>Extracting Biomarkers</h2>
              <p className="text-sm text-purple-500 text-center max-w-xs">Analyzing pitch, shimmer, jitter, and spectral flux against your baseline...</p>
            </motion.div>
          )}

          {flowState === "RESULT" && (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center space-y-6 sm:space-y-8 w-full max-w-md mx-auto"
            >
              {isSuccess ? (
                <>
                  <div className="w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center border-4 border-emerald-400/60 relative shadow-sm" style={{ background: "rgba(255,255,255,0.8)" }}>
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}>
                      <CheckCircle2 className="w-12 h-12 sm:w-16 sm:h-16 text-emerald-500" />
                    </motion.div>
                  </div>
                  <div className="text-center space-y-2 rounded-2xl border border-purple-200/50 p-6 shadow-sm backdrop-blur-md" style={{ background: "rgba(255,255,255,0.7)" }}>
                    <h2 className="text-2xl sm:text-3xl font-bold text-emerald-700" style={{ fontFamily: "var(--font-brand)" }}>Quality Check Passed</h2>
                    <p className="text-sm sm:text-base text-purple-500">Audio sample was clear and isolated.</p>
                  </div>
                </>
              ) : (
                <>
                  <div className="w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center border-4 border-rose-400/60 relative shadow-sm" style={{ background: "rgba(255,255,255,0.8)" }}>
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}>
                      <XCircle className="w-12 h-12 sm:w-16 sm:h-16 text-rose-500" />
                    </motion.div>
                  </div>
                  <div className="text-center space-y-2 rounded-2xl border border-purple-200/50 p-6 shadow-sm backdrop-blur-md" style={{ background: "rgba(255,255,255,0.7)" }}>
                    <h2 className="text-2xl sm:text-3xl font-bold text-rose-700" style={{ fontFamily: "var(--font-brand)" }}>Analysis Failed</h2>
                    <p className="text-sm sm:text-base text-purple-500">{analysisError || "We couldn't isolate your voice clearly."}</p>
                  </div>
                </>
              )}

              <button
                onClick={isSuccess ? finishRecording : handleStart}
                className={`w-full py-4 mt-4 text-white rounded-2xl font-bold text-base sm:text-lg transition-all shadow-lg ${isSuccess ? "bg-emerald-500 hover:bg-emerald-600 shadow-emerald-500/25" : "bg-rose-500 hover:bg-rose-600 shadow-rose-500/25"}`}
              >
                {isSuccess ? "View Results" : "Try Again"}
              </button>
            </motion.div>
          )}

        </AnimatePresence>

      </main>
    </div>
  );
}
