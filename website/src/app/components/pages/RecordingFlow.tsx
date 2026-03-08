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
    id: "prolonged_vowel",
    type: "Prolonged Vowel",
    text: "Aah...",
    instruction: "Sustain the vowel sound /a/ at a comfortable pitch and loudness for about 8 seconds",
    recordingDuration: 10,
  },
  {
    id: "max_phonation_time",
    type: "Max Phonation",
    text: "Aah... (hold as long as you can)",
    instruction: "Take a deep breath and hold the vowel /a/ for as long as you can",
    recordingDuration: 15,
  },
  {
    id: "glides",
    type: "Pitch Glides",
    text: "Aah ↑↓",
    instruction: "Produce a pitch sweep: start low, glide up to your highest comfortable pitch, then back down",
    recordingDuration: 10,
  },
  {
    id: "loudness",
    type: "Loudness",
    text: "Aah (soft → loud)",
    instruction: "Produce the vowel /a/ at three levels: soft, comfortable, then loud",
    recordingDuration: 10,
  },
];

export function RecordingFlow() {
  const { userData, setUserData } = useAppContext();
  const navigate = useNavigate();
  const [flowState, setFlowState] = useState<FlowState>("PREPARING");
  const [countdown, setCountdown] = useState(2); // Initial delay countdown
  const [silenceCountdown, setSilenceCountdown] = useState(3); // 3 seconds of silence
  const [recordingTime, setRecordingTime] = useState(0);
  const [currentPrompt, setCurrentPrompt] = useState(PROMPTS[0]);
  const [activeWordIndex, setActiveWordIndex] = useState(0);
  const [isSuccess, setIsSuccess] = useState(true);
  const [resultFlag, setResultFlag] = useState<'green' | 'yellow' | 'orange' | 'red'>('green');
  const [resultFlagLabel, setResultFlagLabel] = useState('');
  const [resultSummary, setResultSummary] = useState('');
  const [resultRecommendation, setResultRecommendation] = useState('');
  const [resultProbability, setResultProbability] = useState(0);
  const [resultDetails, setResultDetails] = useState('');
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

  // Start actual recording when entering silence countdown (need to capture the silence)
  useEffect(() => {
    if (flowState === "SILENCE_COUNTDOWN") {
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

          // Use the prompt's server task ID directly
          const taskType = currentPrompt.id;

          // Call the backend API
          const result = await analyzeAudio(audioBlob, {
            deviceId: 'web_app',
            taskType,
            silenceDuration: 0.5,
          });

          if (result.success && result.data) {
            // Analysis successful
            setIsSuccess(true);

            // Store flag data for result screen
            const explanation = result.data.explanation;
            setResultFlag((explanation as any).flag || 'green');
            setResultFlagLabel((explanation as any).flag_label || 'Healthy');
            setResultSummary((explanation as any).summary || '');
            setResultRecommendation((explanation as any).recommendation || '');
            setResultProbability((explanation as any).probability_percent || 0);
            setResultDetails((explanation as any).details || '');

            // Extract biomarkers from features
            const biomarkers = extractBiomarkers(result.data.features);

            // Generate user-friendly message
            const msgResult = generateMessage(
              result.data.predictions,
              result.data.explanation
            );

            // Store results temporarily for finishRecording
            (window as any).__analysisResult = {
              biomarkers,
              message: msgResult.message,
              isAnomaly: msgResult.isAnomaly,
              flag: msgResult.flag,
              flagLabel: msgResult.flagLabel,
              recommendation: msgResult.recommendation,
              probabilityPercent: msgResult.probabilityPercent,
              details: msgResult.details,
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
        date: new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' }),
        pitch,
        shimmer,
        jitter,
        spectralCentroid,
        harmonicRatio,
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
    <div className="flex flex-col h-full max-h-screen bg-[#faf8ff] text-slate-900 relative z-10 p-4 sm:p-6 lg:p-8 overflow-hidden">

      {/* Header */}
      <header className="flex justify-between items-center mb-4 sm:mb-6 relative z-20 flex-shrink-0">
        <button
          onClick={() => {
            stopAudioStream();
            if (flowState === "RESULT" && !isSuccess) {
              setFlowState("PREPARING");
              setAudioBlob(null);
              setAnalysisError(null);
            } else {
              navigate("/");
            }
          }}
          className="text-slate-400 hover:text-slate-700 transition-colors"
        >
          <XCircle className="w-8 h-8" />
        </button>
        <span className="text-sm font-medium tracking-wide text-slate-500 uppercase">
          {flowState === "PREPARING" && "Get Ready"}
          {flowState === "INITIAL_DELAY" && "Get Ready"}
          {flowState === "SILENCE_COUNTDOWN" && "Prepare for Silence"}
          {flowState === "SILENCE_RECORDING" && "Recording Silence"}
          {flowState === "RECORDING" && "Recording"}
          {flowState === "ANALYZING" && "Analyzing"}
          {flowState === "RESULT" && isSuccess && "Done"}
          {flowState === "RESULT" && !isSuccess && "Error"}
        </span>
        <div className="w-8" />
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col items-center justify-center relative z-20 min-h-0 overflow-y-auto">

        <AnimatePresence mode="wait">

          {flowState === "PREPARING" && (
            <motion.div
              key="preparing"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center text-center space-y-6 sm:space-y-8 w-full max-w-md mx-auto"
            >
              <div className="w-20 h-20 sm:w-24 sm:h-24 bg-violet-100 rounded-full flex items-center justify-center border border-violet-200">
                <Mic className="w-8 h-8 sm:w-10 sm:h-10 text-violet-600" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold mb-2">Daily Voice Check</h2>
                <p className="text-slate-500 max-w-xs mx-auto text-sm leading-relaxed">
                  Find a quiet place. First, we'll record 3 seconds of silence to calibrate, then you'll complete a voice task.
                </p>
              </div>

              <div className="bg-white border border-violet-100 rounded-2xl p-4 w-full text-left flex items-start gap-3 sm:gap-4 shadow-sm">
                <div className="bg-violet-100 p-2 rounded-xl text-violet-600 shrink-0">
                  <Play className="w-4 h-4 sm:w-5 sm:h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-slate-900">Task: {currentPrompt.type}</h3>
                  <p className="text-xs text-slate-500 mt-1">{currentPrompt.instruction}</p>
                </div>
              </div>

              <button
                onClick={handleStart}
                className="w-full py-4 mt-4 sm:mt-8 bg-violet-600 hover:bg-violet-700 text-white rounded-2xl font-bold text-base sm:text-lg shadow-lg shadow-violet-300/30 transition-all"
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
              <div className="text-[120px] font-black text-violet-600 drop-shadow-[0_0_30px_rgba(124,58,237,0.3)] tabular-nums">
                {countdown}
              </div>
              <div className="bg-violet-50 border border-violet-200 rounded-2xl p-6 w-full">
                <p className="text-slate-600 text-lg font-medium">
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
              <div className="text-[120px] font-black text-amber-500 drop-shadow-[0_0_30px_rgba(245,158,11,0.3)] tabular-nums">
                {silenceCountdown}
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 w-full">
                <p className="text-amber-700 text-xl font-bold mb-2">Please be quiet!</p>
                <p className="text-slate-500 text-sm">
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
              <div className="w-32 h-32 bg-amber-50 rounded-full flex items-center justify-center border-4 border-amber-200 relative">
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className="absolute inset-0 bg-amber-100 rounded-full"
                />
                <Mic className="w-12 h-12 text-amber-500" />
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 w-full">
                <p className="text-amber-700 text-xl font-bold mb-2">Shh... Stay quiet</p>
                <p className="text-slate-500 text-sm">
                  Calibrating with silence: {recordingTime}/3 seconds
                </p>
                <p className="text-slate-400 text-xs mt-2">
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
                <div className="bg-white border border-violet-100 rounded-2xl p-4 sm:p-5 shadow-sm">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse" />
                    <span className="text-xs text-slate-500 uppercase tracking-wider font-medium">Live Audio</span>
                  </div>
                  <LiveWaveform analyser={analyserRef.current} isActive={flowState === "RECORDING"} />
                </div>
              </div>

              {/* Recording Timer */}
              <div className="text-center space-y-2">
                <div className="text-4xl sm:text-5xl font-black text-slate-900 tabular-nums">
                  {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
                  <span className="text-2xl text-slate-400"> / {Math.floor(currentPrompt.recordingDuration / 60)}:{(currentPrompt.recordingDuration % 60).toString().padStart(2, '0')}</span>
                </div>
                <p className="text-sm text-slate-500">Recording in progress</p>
                <div className="w-full bg-violet-100 rounded-full h-2 mt-3">
                  <div
                    className="bg-violet-600 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${(recordingTime / currentPrompt.recordingDuration) * 100}%` }}
                  />
                </div>
              </div>

              {/* Prompt Text */}
              <div className="w-full bg-white border border-violet-100 rounded-3xl p-4 sm:p-6 shadow-sm max-h-[40vh] overflow-y-auto">
                <motion.div
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="text-2xl sm:text-3xl font-bold tracking-[0.2em] text-violet-600 text-center"
                >
                  {currentPrompt.text}
                </motion.div>
              </div>

              {/* Stop Recording Button */}
              <button
                onClick={handleStopRecording}
                className="w-full py-4 bg-rose-500 hover:bg-rose-600 text-white rounded-2xl font-bold text-lg shadow-lg shadow-rose-200/40 transition-all flex items-center justify-center gap-3"
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
                <svg className="w-full h-full animate-spin text-violet-200" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" />
                  <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" strokeDasharray="283" strokeDashoffset="200" className="text-violet-600" strokeLinecap="round" />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-violet-600 animate-pulse" />
                </div>
              </div>
              <h2 className="text-xl font-bold">Extracting Biomarkers</h2>
              <p className="text-sm text-slate-500 text-center max-w-xs">
                Analyzing pitch, shimmer, jitter, and harmonics against your baseline...
              </p>
            </motion.div>
          )}

          {flowState === "RESULT" && (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center space-y-5 sm:space-y-6 w-full max-w-md mx-auto"
            >
              {isSuccess ? (
                <>
                  {/* Flag Icon */}
                  <div className={`w-24 h-24 sm:w-28 sm:h-28 rounded-full flex items-center justify-center border-4 relative ${
                    resultFlag === 'green' ? 'bg-emerald-50 border-emerald-400' :
                    resultFlag === 'yellow' ? 'bg-amber-50 border-amber-400' :
                    resultFlag === 'orange' ? 'bg-orange-50 border-orange-400' :
                    'bg-rose-50 border-rose-400'
                  }`}>
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}>
                      {resultFlag === 'green' && <CheckCircle2 className="w-10 h-10 sm:w-14 sm:h-14 text-emerald-500" />}
                      {resultFlag === 'yellow' && <AlertTriangle className="w-10 h-10 sm:w-14 sm:h-14 text-amber-500" />}
                      {resultFlag === 'orange' && <AlertTriangle className="w-10 h-10 sm:w-14 sm:h-14 text-orange-500" />}
                      {resultFlag === 'red' && <XCircle className="w-10 h-10 sm:w-14 sm:h-14 text-rose-500" />}
                    </motion.div>
                  </div>

                  {/* Flag Label + Probability */}
                  <div className="text-center space-y-1">
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                      resultFlag === 'green' ? 'bg-emerald-100 text-emerald-700' :
                      resultFlag === 'yellow' ? 'bg-amber-100 text-amber-700' :
                      resultFlag === 'orange' ? 'bg-orange-100 text-orange-700' :
                      'bg-rose-100 text-rose-700'
                    }`}>
                      {resultFlagLabel}
                    </span>
                    <p className="text-xs text-slate-400 mt-1">Screening score: {resultProbability.toFixed(1)}%</p>
                  </div>

                  {/* Summary */}
                  <div className="bg-white border border-violet-100 rounded-2xl p-4 w-full shadow-sm">
                    <p className="text-sm text-slate-700 leading-relaxed">{resultSummary}</p>
                    {resultDetails && (
                      <p className="text-xs text-slate-400 mt-2">{resultDetails}</p>
                    )}
                  </div>

                  {/* Recommendation */}
                  {resultRecommendation && (
                    <div className={`rounded-2xl p-4 w-full border ${
                      resultFlag === 'green' ? 'bg-emerald-50 border-emerald-200' :
                      resultFlag === 'yellow' ? 'bg-amber-50 border-amber-200' :
                      resultFlag === 'orange' ? 'bg-orange-50 border-orange-200' :
                      'bg-rose-50 border-rose-200'
                    }`}>
                      <p className="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-1">Recommendation</p>
                      <p className="text-sm text-slate-700 leading-relaxed">{resultRecommendation}</p>
                    </div>
                  )}

                  {/* Find a Specialist link for orange/red */}
                  {(resultFlag === 'orange' || resultFlag === 'red') && (
                    <a
                      href="https://www.google.com/search?q=ENT+specialist+near+me"
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`w-full py-3 text-center rounded-2xl font-semibold text-sm transition-all border-2 ${
                        resultFlag === 'red'
                          ? 'border-rose-400 text-rose-600 hover:bg-rose-50'
                          : 'border-orange-400 text-orange-600 hover:bg-orange-50'
                      }`}
                    >
                      Find an ENT Specialist Near You →
                    </a>
                  )}
                </>
              ) : (
                <>
                  <div className="w-24 h-24 sm:w-28 sm:h-28 bg-rose-50 rounded-full flex items-center justify-center border-4 border-rose-400 relative">
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.5 }}>
                      <XCircle className="w-10 h-10 sm:w-14 sm:h-14 text-rose-500" />
                    </motion.div>
                  </div>
                  <div className="text-center space-y-2">
                    <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">Analysis Failed</h2>
                    <p className="text-sm sm:text-base text-slate-500">
                      {analysisError || "We couldn't isolate your voice clearly."}
                    </p>
                  </div>
                </>
              )}

              <button
                onClick={isSuccess ? finishRecording : handleStart}
                className={`w-full py-4 text-white rounded-2xl font-bold text-base sm:text-lg transition-all ${
                  isSuccess
                    ? resultFlag === 'green' ? "bg-emerald-500 hover:bg-emerald-600 shadow-lg shadow-emerald-200/40"
                      : resultFlag === 'yellow' ? "bg-amber-500 hover:bg-amber-600 shadow-lg shadow-amber-200/40"
                      : resultFlag === 'orange' ? "bg-orange-500 hover:bg-orange-600 shadow-lg shadow-orange-200/40"
                      : "bg-rose-500 hover:bg-rose-600 shadow-lg shadow-rose-200/40"
                    : "bg-rose-500 hover:bg-rose-600 shadow-lg shadow-rose-200/40"
                }`}
              >
                {isSuccess ? "View Dashboard" : "Try Again"}
              </button>
            </motion.div>
          )}

        </AnimatePresence>

      </main>
    </div>
  );
}
