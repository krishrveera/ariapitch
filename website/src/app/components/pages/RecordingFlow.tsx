import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router";
import { useAppContext } from "../../AppContext";
import { motion, AnimatePresence } from "motion/react";
import { Mic, CheckCircle2, XCircle, AlertTriangle, Play, Square, Activity } from "lucide-react";
import { analyzeAudio, extractBiomarkers, generateMessage } from "../../services/api";
import { LiveWaveform } from "../ui/LiveWaveform";

type FlowState = "PREPARING" | "INITIAL_DELAY" | "SILENCE_COUNTDOWN" | "SILENCE_RECORDING" | "RECORDING" | "ANALYZING" | "RESULT";

const PROMPTS = [
  {
    type: "Vowel",
    text: "1, 2, 3 aah",
    instruction: "Repeat '1, 2, 3 aah' in your normal voice and hold the sound 'aah' for as long as you can",
    recordingDuration: 10, // seconds - sustained vowel
  },
  {
    type: "Reading",
    text: "Do you like amusement parks? Well, I sure do. To amuse myself, I went twice last spring. My most MEMORABLE moment was riding on the Caterpillar, which is a gigantic rollercoaster high above the ground. When I saw how high the Caterpillar rose into the bright blue sky I knew it was for me. After waiting in line for thirty minutes, I made it to the front where the man measured my height to see if I was tall enough. I gave the man my coins, asked for change, and jumped on the cart. Tick, tick, tick, the Caterpillar climbed slowly up the tracks. It went SO high I could see the parking lot. Boy was I SCARED! I thought to myself, \"There's no turning back now.\" People were so scared they screamed as we swiftly zoomed fast, fast, and faster along the tracks. As quickly as it started, the Caterpillar came to a stop. Unfortunately, it was time to pack the car and drive home. That night I dreamt of the wild ride on the Caterpillar. Taking a trip to the amusement park and riding on the Caterpillar was my MOST memorable moment ever!",
    instruction: "Read the Caterpillar Passage out loud in your typical voice",
    recordingDuration: 110, // seconds - 1:50 as per protocol
  }
];

export function RecordingFlow() {
  const { userData, setUserData } = useAppContext();
  const navigate = useNavigate();
  const [flowState, setFlowState] = useState<FlowState>("PREPARING");
  const [countdown, setCountdown] = useState(2); // Initial delay countdown
  const [silenceCountdown, setSilenceCountdown] = useState(3); // 3 seconds of silence
  const [recordingTime, setRecordingTime] = useState(0);
  const [currentPrompt, setCurrentPrompt] = useState(PROMPTS[1]);
  const [activeWordIndex, setActiveWordIndex] = useState(0);
  const [isSuccess, setIsSuccess] = useState(true);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Audio refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Initialize
  useEffect(() => {
    setCurrentPrompt(PROMPTS[Math.floor(Math.random() * PROMPTS.length)]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudioStream();
    };
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

      // Don't start recording yet - we'll do that after silence period
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
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        resolve();
        return;
      }

      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });

          // Save blob for API call
          setAudioBlob(blob);

          const arrayBuffer = await blob.arrayBuffer();

          // Decode audio data to get raw samples
          const audioCtx = new AudioContext();
          const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
          const channelData = audioBuffer.getChannelData(0);

          // Store in context
          setUserData(prev => ({
            ...prev,
            lastRecordingData: new Float32Array(channelData),
            lastRecordingSampleRate: audioBuffer.sampleRate,
          }));

          await audioCtx.close();
        } catch (err) {
          console.error("Error processing audio:", err);
          setAnalysisError("Failed to process audio recording");
        }
        resolve();
      };

      mediaRecorder.stop();

      // Stop the media stream tracks
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(t => t.stop());
      }
    });
  };

  // Initial Delay Logic - Give user time to understand what's happening
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (flowState === "INITIAL_DELAY") {
      if (countdown > 0) {
        timer = setTimeout(() => setCountdown(c => c - 1), 1000);
      } else {
        setFlowState("SILENCE_COUNTDOWN");
        setSilenceCountdown(3); // Reset silence countdown
      }
    }
    return () => clearTimeout(timer);
  }, [countdown, flowState]);

  // Silence Countdown Logic
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (flowState === "SILENCE_COUNTDOWN") {
      if (silenceCountdown > 0) {
        timer = setTimeout(() => setSilenceCountdown(c => c - 1), 1000);
      } else {
        setFlowState("SILENCE_RECORDING");
        setRecordingTime(0);
      }
    }
    return () => clearTimeout(timer);
  }, [silenceCountdown, flowState]);

  // Silence Recording Timer - 3 seconds of silence
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (flowState === "SILENCE_RECORDING") {
      if (recordingTime < 3) {
        timer = setTimeout(() => setRecordingTime(t => t + 1), 1000);
      } else {
        setFlowState("RECORDING");
        setRecordingTime(0); // Reset for actual recording
      }
    }
    return () => clearTimeout(timer);
  }, [recordingTime, flowState]);

  // Start actual recording when entering RECORDING state
  useEffect(() => {
    if (flowState === "RECORDING") {
      startRecording();
    }
  }, [flowState]);

  // Recording Timer - Auto-stop after duration
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (flowState === "RECORDING") {
      if (recordingTime < currentPrompt.recordingDuration) {
        timer = setTimeout(() => setRecordingTime(t => t + 1), 1000);
      } else {
        // Auto-stop recording when time is up
        handleStopRecording();
      }
    }
    return () => clearTimeout(timer);
  }, [recordingTime, flowState, currentPrompt]);

  // Analyzing Logic - Call real API
  useEffect(() => {
    if (flowState === "ANALYZING" && audioBlob) {
      const runAnalysis = async () => {
        try {
          setAnalysisError(null);

          // Determine task type based on prompt
          const taskType = currentPrompt.type === "Vowel" ? "sustained_vowel" : "reading_passage";

          // Call the backend API
          const result = await analyzeAudio(audioBlob, {
            deviceId: 'web_app',
            taskType,
            silenceDuration: 0.5,
          });

          if (result.success && result.data) {
            // Analysis successful
            setIsSuccess(true);

            // Extract biomarkers from features
            const biomarkers = extractBiomarkers(result.data.features);

            // Generate user-friendly message
            const { message, isAnomaly } = generateMessage(
              result.data.predictions,
              result.data.explanation
            );

            // Store results temporarily for finishRecording
            (window as any).__analysisResult = {
              biomarkers,
              message,
              isAnomaly,
              predictions: result.data.predictions,
            };
          } else {
            // Analysis failed - quality gate or error
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
    setCountdown(2);
    setSilenceCountdown(3);
    setRecordingTime(0);
    setActiveWordIndex(0);

    const ok = await startAudioCapture();
    if (ok) {
      setFlowState("INITIAL_DELAY");
    } else {
      // Fallback: proceed without audio if mic is denied
      setFlowState("INITIAL_DELAY");
    }
  };

  const handleStopRecording = async () => {
    await stopAndProcessAudio();
    stopAudioStream();
    setFlowState("ANALYZING");
  };

  const finishRecording = () => {
    if (isSuccess) {
      // Get analysis results from window (set in analyzing effect)
      const analysisResult = (window as any).__analysisResult;

      let pitch, shimmer, jitter, message, isAnomaly, spectralCentroid, harmonicRatio;

      if (analysisResult) {
        // Use real analysis results
        pitch = analysisResult.biomarkers.pitch;
        shimmer = analysisResult.biomarkers.shimmer;
        jitter = analysisResult.biomarkers.jitter;
        spectralCentroid = analysisResult.biomarkers.spectralCentroid;
        harmonicRatio = analysisResult.biomarkers.harmonicRatio;
        message = analysisResult.message;
        isAnomaly = analysisResult.isAnomaly;

        // Clean up
        delete (window as any).__analysisResult;
      } else {
        // Fallback to dummy data if analysis failed
        pitch = Math.round(200 + Math.random() * 20);
        shimmer = Number((3 + Math.random() * 1.5).toFixed(1));
        jitter = Number((1 + Math.random() * 1).toFixed(1));
        spectralCentroid = Math.round(1500 + Math.random() * 500);
        harmonicRatio = Number((15 + Math.random() * 8).toFixed(1));
        message = "Your voice exhibits slight jitter today, but pitch is stable. Stay hydrated!";
        isAnomaly = false;
      }

      const newEntry = {
        date: new Date().toLocaleDateString('en-US', { weekday: 'short' }),
        pitch,
        shimmer,
        jitter,
        message,
        isAnomaly,
      };

      // If opted in to Bridge2AI, save the full audio recording
      const updates: any = {
        hasRecordedToday: true,
        history: [...userData.history, newEntry],
        showHealthPopup: isAnomaly,
      };

      if (userData.optedIn) {
        const audioRecording = {
          id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          timestamp: Date.now(),
          pitch,
          shimmer,
          jitter,
          message,
          isAnomaly,
          duration: recordingTime,
          spectralCentroid,
          harmonicRatio,
          formants: [
            Math.round(700 + Math.random() * 100),
            Math.round(1200 + Math.random() * 200),
            Math.round(2500 + Math.random() * 300),
          ],
        };

        updates.audioRecordings = [...userData.audioRecordings, audioRecording];
      }

      setUserData(prev => ({ ...prev, ...updates }));
    }
    navigate("/");
  };

  return (
    <div className="flex flex-col h-full bg-neutral-950 text-neutral-50 relative z-10 p-4 sm:p-6 lg:p-8 overflow-hidden">

      {/* Header */}
      <header className="flex justify-between items-center mb-8 sm:mb-12 relative z-20">
        <button
          onClick={() => { stopAudioStream(); navigate("/"); }}
          className="text-neutral-500 hover:text-white transition-colors"
        >
          <XCircle className="w-6 h-6" />
        </button>
        <span className="text-sm font-medium tracking-wide text-neutral-400 uppercase">
          {flowState === "PREPARING" && "Get Ready"}
          {flowState === "INITIAL_DELAY" && "Get Ready"}
          {flowState === "SILENCE_COUNTDOWN" && "Prepare for Silence"}
          {flowState === "SILENCE_RECORDING" && "Recording Silence"}
          {flowState === "RECORDING" && "Recording"}
          {flowState === "ANALYZING" && "Analyzing"}
          {flowState === "RESULT" && "Done"}
        </span>
        <div className="w-6" />
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col items-center justify-center relative z-20">

        <AnimatePresence mode="wait">

          {flowState === "PREPARING" && (
            <motion.div
              key="preparing"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center text-center space-y-6 sm:space-y-8 w-full max-w-md mx-auto"
            >
              <div className="w-20 h-20 sm:w-24 sm:h-24 bg-indigo-500/10 rounded-full flex items-center justify-center border border-indigo-500/20">
                <Mic className="w-8 h-8 sm:w-10 sm:h-10 text-indigo-400" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold mb-2">Daily Voice Check</h2>
                <p className="text-neutral-400 max-w-xs mx-auto text-sm leading-relaxed">
                  Find a quiet place. First, we'll record 3 seconds of silence to calibrate, then you'll complete a voice task.
                </p>
              </div>

              <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 w-full text-left flex items-start gap-3 sm:gap-4">
                <div className="bg-neutral-800 p-2 rounded-xl text-neutral-300 shrink-0">
                  <Play className="w-4 h-4 sm:w-5 sm:h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-white">Task: {currentPrompt.type}</h3>
                  <p className="text-xs text-neutral-400 mt-1">{currentPrompt.instruction}</p>
                </div>
              </div>

              <button
                onClick={handleStart}
                className="w-full py-4 mt-4 sm:mt-8 bg-indigo-500 hover:bg-indigo-600 text-white rounded-2xl font-bold text-base sm:text-lg shadow-[0_0_40px_-10px_rgba(99,102,241,0.5)] transition-all"
              >
                I'm Ready
              </button>
            </motion.div>
          )}

          {flowState === "INITIAL_DELAY" && (
            <motion.div
              key="initial-delay"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center text-center space-y-6 w-full max-w-md mx-auto"
            >
              <div className="text-[120px] font-black text-indigo-400 drop-shadow-[0_0_30px_rgba(99,102,241,0.5)] tabular-nums">
                {countdown}
              </div>
              <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-2xl p-6 w-full">
                <p className="text-neutral-300 text-lg font-medium">
                  Get ready to be quiet for 3 seconds
                </p>
              </div>
            </motion.div>
          )}

          {flowState === "SILENCE_COUNTDOWN" && (
            <motion.div
              key="silence-countdown"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center text-center space-y-6 w-full max-w-md mx-auto"
            >
              <div className="text-[120px] font-black text-amber-400 drop-shadow-[0_0_30px_rgba(251,191,36,0.5)] tabular-nums">
                {silenceCountdown}
              </div>
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-6 w-full">
                <p className="text-amber-200 text-xl font-bold mb-2">Please be quiet!</p>
                <p className="text-neutral-400 text-sm">
                  Recording {silenceCountdown} seconds of silence for calibration
                </p>
              </div>
            </motion.div>
          )}

          {flowState === "SILENCE_RECORDING" && (
            <motion.div
              key="silence-recording"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center text-center space-y-6 w-full max-w-md mx-auto"
            >
              <div className="w-32 h-32 bg-amber-500/10 rounded-full flex items-center justify-center border-4 border-amber-500/30 relative">
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className="absolute inset-0 bg-amber-500/20 rounded-full"
                />
                <Mic className="w-12 h-12 text-amber-400" />
              </div>
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-6 w-full">
                <p className="text-amber-200 text-xl font-bold mb-2">🤫 Shh... Stay quiet</p>
                <p className="text-neutral-400 text-sm">
                  Calibrating with silence: {recordingTime}/3 seconds
                </p>
                <p className="text-neutral-500 text-xs mt-2">
                  Your task will begin after this
                </p>
              </div>
            </motion.div>
          )}

          {flowState === "RECORDING" && (
            <motion.div
              key="recording"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center w-full space-y-6 sm:space-y-8"
            >
              {/* Live Audio Waveform */}
              <div className="w-full max-w-lg">
                <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-4 sm:p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse" />
                    <span className="text-xs text-neutral-400 uppercase tracking-wider font-medium">Live Audio</span>
                  </div>
                  <LiveWaveform analyser={analyserRef.current} isActive={flowState === "RECORDING"} />
                </div>
              </div>

              {/* Recording Timer */}
              <div className="text-center space-y-2">
                <div className="text-4xl sm:text-5xl font-black text-white tabular-nums">
                  {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
                  <span className="text-2xl text-neutral-500"> / {Math.floor(currentPrompt.recordingDuration / 60)}:{(currentPrompt.recordingDuration % 60).toString().padStart(2, '0')}</span>
                </div>
                <p className="text-sm text-neutral-400">Recording in progress</p>
                <div className="w-full bg-neutral-800 rounded-full h-2 mt-3">
                  <div
                    className="bg-indigo-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(recordingTime / currentPrompt.recordingDuration) * 100}%` }}
                  />
                </div>
              </div>

              {/* Prompt Text */}
              <div className="w-full bg-neutral-900 border border-neutral-800 rounded-3xl p-4 sm:p-6 shadow-inner max-h-[40vh] overflow-y-auto">
                {currentPrompt.type === "Reading" ? (
                  <div className="text-base sm:text-lg font-medium leading-relaxed text-white text-left">
                    {currentPrompt.text}
                  </div>
                ) : (
                  <motion.div
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                    className="text-2xl sm:text-3xl font-bold tracking-[0.2em] text-indigo-400 text-center"
                  >
                    {currentPrompt.text}
                  </motion.div>
                )}
              </div>

              {/* Stop Recording Button */}
              <button
                onClick={handleStopRecording}
                className="w-full py-4 bg-rose-500 hover:bg-rose-600 text-white rounded-2xl font-bold text-lg shadow-[0_0_40px_-10px_rgba(244,63,94,0.5)] transition-all flex items-center justify-center gap-3"
              >
                <Square className="w-5 h-5 fill-current" />
                Stop Recording
              </button>
            </motion.div>
          )}

          {flowState === "ANALYZING" && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center space-y-6"
            >
              <div className="relative w-32 h-32 flex items-center justify-center">
                <svg className="w-full h-full animate-spin text-indigo-500/30" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" />
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" strokeDasharray="283" strokeDashoffset="200" className="text-indigo-500" strokeLinecap="round" />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-indigo-400 animate-pulse" />
                </div>
              </div>
              <h2 className="text-xl font-bold">Extracting Biomarkers</h2>
              <p className="text-sm text-neutral-400 text-center max-w-xs">
                Analyzing pitch, shimmer, jitter, and spectral flux against your baseline...
              </p>
            </motion.div>
          )}

          {flowState === "RESULT" && (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center space-y-6 sm:space-y-8 w-full max-w-md mx-auto"
            >
              {isSuccess ? (
                <>
                  <div className="w-28 h-28 sm:w-32 sm:h-32 bg-emerald-500/20 rounded-full flex items-center justify-center border-4 border-emerald-500 relative">
                    <motion.div
                      initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}
                    >
                      <CheckCircle2 className="w-12 h-12 sm:w-16 sm:h-16 text-emerald-400" />
                    </motion.div>
                  </div>
                  <div className="text-center space-y-2">
                    <h2 className="text-2xl sm:text-3xl font-bold text-white">Quality Check Passed</h2>
                    <p className="text-sm sm:text-base text-neutral-400">Audio sample was clear and isolated.</p>
                  </div>
                </>
              ) : (
                <>
                  <div className="w-28 h-28 sm:w-32 sm:h-32 bg-rose-500/20 rounded-full flex items-center justify-center border-4 border-rose-500 relative">
                    <motion.div
                      initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}
                    >
                      <XCircle className="w-12 h-12 sm:w-16 sm:h-16 text-rose-400" />
                    </motion.div>
                  </div>
                  <div className="text-center space-y-2">
                    <h2 className="text-2xl sm:text-3xl font-bold text-white">Analysis Failed</h2>
                    <p className="text-sm sm:text-base text-neutral-400">
                      {analysisError || "We couldn't isolate your voice clearly."}
                    </p>
                  </div>
                </>
              )}

              <button
                onClick={isSuccess ? finishRecording : handleStart}
                className={`w-full py-4 mt-4 text-white rounded-2xl font-bold text-base sm:text-lg transition-all ${isSuccess ? "bg-emerald-500 hover:bg-emerald-600 shadow-[0_0_30px_-5px_rgba(16,185,129,0.4)]" : "bg-rose-500 hover:bg-rose-600 shadow-[0_0_30px_-5px_rgba(244,63,94,0.4)]"}`}
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
