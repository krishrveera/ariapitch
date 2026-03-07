import { useState, useRef, useEffect, useCallback } from "react";

// ============================================================
// CONFIG
// ============================================================
const RAINBOW_PASSAGE = `When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.`;

const MIN_DURATION_SEC = 8;
const MAX_DURATION_SEC = 60;
const CLIPPING_THRESHOLD = 0.98;
const SILENCE_THRESHOLD = 0.01;
const SILENCE_MAX_RATIO = 0.7;

// ============================================================
// QUALITY ANALYSIS (runs on raw audio buffer)
// ============================================================
function analyzeAudioQuality(audioBuffer) {
  const data = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const duration = audioBuffer.duration;
  const issues = [];
  let passed = true;

  // Duration check
  if (duration < MIN_DURATION_SEC) {
    issues.push({ type: "error", msg: `Recording too short (${duration.toFixed(1)}s). Please read the full passage.` });
    passed = false;
  }
  if (duration > MAX_DURATION_SEC) {
    issues.push({ type: "warning", msg: `Recording is quite long (${duration.toFixed(1)}s). That's okay, but unusual.` });
  }

  // Clipping check
  let clipCount = 0;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) > CLIPPING_THRESHOLD) clipCount++;
  }
  const clipRatio = clipCount / data.length;
  if (clipRatio > 0.005) {
    issues.push({ type: "error", msg: "Audio clipping detected. Try moving further from the microphone." });
    passed = false;
  } else if (clipRatio > 0.001) {
    issues.push({ type: "warning", msg: "Minor clipping detected. Recording is usable but not ideal." });
  }

  // Silence detection
  const frameSize = Math.floor(sampleRate * 0.025); // 25ms frames
  let silentFrames = 0;
  let totalFrames = 0;
  for (let i = 0; i < data.length - frameSize; i += frameSize) {
    let rms = 0;
    for (let j = i; j < i + frameSize; j++) {
      rms += data[j] * data[j];
    }
    rms = Math.sqrt(rms / frameSize);
    totalFrames++;
    if (rms < SILENCE_THRESHOLD) silentFrames++;
  }
  const silenceRatio = silentFrames / totalFrames;
  if (silenceRatio > SILENCE_MAX_RATIO) {
    issues.push({ type: "error", msg: `Too much silence (${(silenceRatio * 100).toFixed(0)}%). Make sure you're reading aloud.` });
    passed = false;
  }

  // RMS level check
  let totalRms = 0;
  for (let i = 0; i < data.length; i++) {
    totalRms += data[i] * data[i];
  }
  totalRms = Math.sqrt(totalRms / data.length);
  if (totalRms < 0.005) {
    issues.push({ type: "error", msg: "Volume too low. Please speak closer to the microphone." });
    passed = false;
  }

  if (passed && issues.length === 0) {
    issues.push({ type: "success", msg: "Audio quality looks good." });
  }

  return {
    passed,
    issues,
    stats: {
      duration: duration.toFixed(1),
      clipRatio: (clipRatio * 100).toFixed(3),
      silenceRatio: (silenceRatio * 100).toFixed(1),
      rmsLevel: (totalRms * 100).toFixed(2),
      sampleRate,
    },
  };
}

// ============================================================
// WAVEFORM VISUALIZATION
// ============================================================
function WaveformCanvas({ analyserRef, isRecording }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    if (!isRecording || !analyserRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      ctx.fillStyle = "#0a1a1f";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#02C39A";
      ctx.beginPath();
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [isRecording, analyserRef]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={100}
      style={{
        width: "100%",
        height: "80px",
        borderRadius: "8px",
        border: "1px solid #1a3a40",
      }}
    />
  );
}

// ============================================================
// TIMER
// ============================================================
function RecordingTimer({ isRecording, startTime }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!isRecording) { setElapsed(0); return; }
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 200);
    return () => clearInterval(interval);
  }, [isRecording, startTime]);
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return (
    <span style={{ fontVariantNumeric: "tabular-nums", fontFamily: "'DM Mono', monospace", fontSize: "28px", color: isRecording ? "#02C39A" : "#4a6670", letterSpacing: "2px" }}>
      {String(mins).padStart(2, "0")}:{String(secs).padStart(2, "0")}
    </span>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [screen, setScreen] = useState("landing"); // landing | record | quality | results
  const [isRecording, setIsRecording] = useState(false);
  const [startTime, setStartTime] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [qualityResult, setQualityResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  // ---- RECORDING LOGIC ----
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: 44100 },
      });
      streamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
      audioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setAnalyzing(true);
        setScreen("quality");

        // Decode and analyze
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const result = analyzeAudioQuality(audioBuffer);
        setQualityResult(result);
        setAnalyzing(false);
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setStartTime(Date.now());
    } catch (err) {
      alert("Microphone access denied. Please allow microphone permissions and try again.");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
    setIsRecording(false);
  }, []);

  const resetRecording = () => {
    setAudioBlob(null);
    setQualityResult(null);
    setScreen("record");
  };

  // ---- STYLES ----
  const styles = {
    app: { minHeight: "100vh", background: "linear-gradient(165deg, #040e12 0%, #0a1a1f 40%, #0d2129 100%)", color: "#e0ede8", fontFamily: "'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif", display: "flex", flexDirection: "column", alignItems: "center" },
    container: { width: "100%", maxWidth: "640px", padding: "20px 24px", flex: 1, display: "flex", flexDirection: "column" },
    header: { display: "flex", alignItems: "center", gap: "12px", padding: "16px 0", marginBottom: "8px", borderBottom: "1px solid #1a3a40" },
    logo: { width: "36px", height: "36px", borderRadius: "10px", background: "linear-gradient(135deg, #028090, #02C39A)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "18px", fontWeight: 700, color: "#fff" },
    brandName: { fontSize: "18px", fontWeight: 600, letterSpacing: "-0.3px", color: "#e0ede8" },
    tagline: { fontSize: "11px", color: "#4a7a6a", fontWeight: 400, marginTop: "2px" },
    h1: { fontSize: "32px", fontWeight: 700, letterSpacing: "-0.8px", lineHeight: 1.2, color: "#fff", margin: "24px 0 12px" },
    subtitle: { fontSize: "15px", color: "#6a9a8a", lineHeight: 1.6, margin: "0 0 32px" },
    card: { background: "rgba(10, 35, 42, 0.7)", border: "1px solid #1a3a40", borderRadius: "12px", padding: "24px", marginBottom: "16px", backdropFilter: "blur(10px)" },
    passageBox: { background: "#060f14", border: "1px solid #1a3a40", borderRadius: "10px", padding: "20px", fontSize: "16px", lineHeight: 1.8, color: "#c0d8d0", fontFamily: "'Georgia', serif", marginBottom: "20px" },
    btnPrimary: { width: "100%", padding: "16px", background: "linear-gradient(135deg, #028090, #02C39A)", color: "#fff", border: "none", borderRadius: "10px", fontSize: "16px", fontWeight: 600, cursor: "pointer", letterSpacing: "0.3px", transition: "opacity 0.2s" },
    btnStop: { width: "100%", padding: "16px", background: "linear-gradient(135deg, #dc2626, #ef4444)", color: "#fff", border: "none", borderRadius: "10px", fontSize: "16px", fontWeight: 600, cursor: "pointer" },
    btnOutline: { width: "100%", padding: "14px", background: "transparent", color: "#02C39A", border: "1px solid #02C39A", borderRadius: "10px", fontSize: "15px", fontWeight: 500, cursor: "pointer", marginTop: "10px" },
    btnDisabled: { width: "100%", padding: "16px", background: "#1a2a30", color: "#4a6a70", border: "1px solid #1a3a40", borderRadius: "10px", fontSize: "16px", fontWeight: 600, cursor: "not-allowed" },
    badge: { display: "inline-block", padding: "4px 10px", borderRadius: "6px", fontSize: "11px", fontWeight: 600, letterSpacing: "0.5px", textTransform: "uppercase" },
    issueRow: { display: "flex", alignItems: "flex-start", gap: "10px", padding: "10px 0", borderBottom: "1px solid #1a3040" },
    statGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: "16px" },
    statCard: { background: "#060f14", borderRadius: "8px", padding: "14px", textAlign: "center" },
    statValue: { fontSize: "22px", fontWeight: 700, color: "#02C39A", fontFamily: "'DM Mono', monospace" },
    statLabel: { fontSize: "11px", color: "#4a7a6a", marginTop: "4px", textTransform: "uppercase", letterSpacing: "0.5px" },
    recordingPulse: { width: "12px", height: "12px", borderRadius: "50%", background: "#ef4444", animation: "pulse 1.2s ease-in-out infinite" },
    footer: { textAlign: "center", padding: "20px", color: "#2a4a4a", fontSize: "12px" },
  };

  // ---- SCREENS ----

  // LANDING
  const LandingScreen = () => (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.logo}>V</div>
        <div>
          <div style={styles.brandName}>VoiceSignal</div>
          <div style={styles.tagline}>Bridge2AI Voice Analysis</div>
        </div>
      </div>

      <h1 style={styles.h1}>One sentence.<br/>One signal.</h1>
      <p style={styles.subtitle}>
        Read a short passage into your microphone. We extract acoustic biomarkers and compare them against
        the Bridge2AI Voice dataset — 833 clinically labeled participants — to generate a voice health profile.
      </p>

      <div style={styles.card}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: "#02C39A", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "1px" }}>How it works</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {[
            { num: "01", title: "Record", desc: "Read the Rainbow Passage aloud (~20 seconds)" },
            { num: "02", title: "Analyze", desc: "We extract pitch, harmonics, and voice quality features" },
            { num: "03", title: "Compare", desc: "Your profile is matched against 833 clinical voice recordings" },
          ].map((step) => (
            <div key={step.num} style={{ display: "flex", gap: "14px", alignItems: "flex-start" }}>
              <div style={{ fontSize: "12px", fontWeight: 700, color: "#028090", fontFamily: "'DM Mono', monospace", minWidth: "28px", paddingTop: "2px" }}>{step.num}</div>
              <div>
                <div style={{ fontSize: "14px", fontWeight: 600, color: "#e0ede8" }}>{step.title}</div>
                <div style={{ fontSize: "13px", color: "#6a9a8a", marginTop: "2px" }}>{step.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ ...styles.card, background: "rgba(2, 128, 144, 0.08)", borderColor: "#0a4a50" }}>
        <div style={{ fontSize: "12px", color: "#6a9a8a", lineHeight: 1.6 }}>
          <strong style={{ color: "#02C39A" }}>Important:</strong> This is a research tool, not a medical device. Results are informational and should be discussed with a healthcare provider. No audio is stored.
        </div>
      </div>

      <button style={styles.btnPrimary} onClick={() => setScreen("record")}>
        Begin Recording →
      </button>

      <div style={styles.footer}>
        Built on Bridge2AI Voice v3.0 · Cornell AI Health 2026
      </div>
    </div>
  );

  // RECORDING
  const RecordScreen = () => (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.logo}>V</div>
        <div>
          <div style={styles.brandName}>VoiceSignal</div>
          <div style={styles.tagline}>Recording</div>
        </div>
        {isRecording && (
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "8px" }}>
            <div style={styles.recordingPulse} />
            <span style={{ fontSize: "13px", color: "#ef4444", fontWeight: 500 }}>REC</span>
          </div>
        )}
      </div>

      <div style={{ marginBottom: "16px" }}>
        <div style={{ fontSize: "13px", fontWeight: 600, color: "#02C39A", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "1px" }}>
          Read this passage aloud
        </div>
        <div style={styles.passageBox}>{RAINBOW_PASSAGE}</div>
      </div>

      {isRecording && (
        <div style={{ ...styles.card, textAlign: "center", padding: "16px 24px" }}>
          <RecordingTimer isRecording={isRecording} startTime={startTime} />
          <div style={{ marginTop: "12px" }}>
            <WaveformCanvas analyserRef={analyserRef} isRecording={isRecording} />
          </div>
        </div>
      )}

      <div style={{ marginTop: "auto", paddingTop: "20px" }}>
        {!isRecording ? (
          <button style={styles.btnPrimary} onClick={startRecording}>
            🎙  Start Recording
          </button>
        ) : (
          <button style={styles.btnStop} onClick={stopRecording}>
            ◼  Stop Recording
          </button>
        )}
        {!isRecording && (
          <button style={styles.btnOutline} onClick={() => setScreen("landing")}>
            ← Back
          </button>
        )}
      </div>
    </div>
  );

  // QUALITY GATE
  const QualityScreen = () => (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.logo}>V</div>
        <div>
          <div style={styles.brandName}>VoiceSignal</div>
          <div style={styles.tagline}>Quality Check</div>
        </div>
      </div>

      {analyzing ? (
        <div style={{ ...styles.card, textAlign: "center", padding: "48px 24px" }}>
          <div style={{ fontSize: "28px", marginBottom: "16px" }}>🔍</div>
          <div style={{ fontSize: "16px", color: "#6a9a8a" }}>Analyzing audio quality...</div>
        </div>
      ) : qualityResult ? (
        <>
          <div style={{ ...styles.card, borderColor: qualityResult.passed ? "#0a4a30" : "#4a1a1a" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
              <div style={{ fontSize: "24px" }}>{qualityResult.passed ? "✓" : "✗"}</div>
              <div>
                <div style={{ fontSize: "16px", fontWeight: 600, color: qualityResult.passed ? "#02C39A" : "#ef4444" }}>
                  {qualityResult.passed ? "Quality Check Passed" : "Quality Issues Detected"}
                </div>
                <div style={{ fontSize: "12px", color: "#6a9a8a", marginTop: "2px" }}>
                  {qualityResult.passed ? "Your recording meets the analysis requirements." : "Please re-record to get accurate results."}
                </div>
              </div>
            </div>

            {qualityResult.issues.map((issue, i) => (
              <div key={i} style={styles.issueRow}>
                <span style={{
                  ...styles.badge,
                  background: issue.type === "error" ? "rgba(239,68,68,0.15)" : issue.type === "warning" ? "rgba(245,158,11,0.15)" : "rgba(2,195,154,0.15)",
                  color: issue.type === "error" ? "#ef4444" : issue.type === "warning" ? "#f59e0b" : "#02C39A",
                }}>
                  {issue.type}
                </span>
                <span style={{ fontSize: "13px", color: "#c0d8d0", lineHeight: 1.5 }}>{issue.msg}</span>
              </div>
            ))}
          </div>

          <div style={styles.card}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "#4a7a6a", marginBottom: "8px", textTransform: "uppercase", letterSpacing: "1px" }}>Recording Stats</div>
            <div style={styles.statGrid}>
              <div style={styles.statCard}>
                <div style={styles.statValue}>{qualityResult.stats.duration}s</div>
                <div style={styles.statLabel}>Duration</div>
              </div>
              <div style={styles.statCard}>
                <div style={styles.statValue}>{qualityResult.stats.sampleRate / 1000}k</div>
                <div style={styles.statLabel}>Sample Rate</div>
              </div>
              <div style={styles.statCard}>
                <div style={styles.statValue}>{qualityResult.stats.silenceRatio}%</div>
                <div style={styles.statLabel}>Silence Ratio</div>
              </div>
              <div style={styles.statCard}>
                <div style={styles.statValue}>{qualityResult.stats.rmsLevel}%</div>
                <div style={styles.statLabel}>RMS Level</div>
              </div>
            </div>
          </div>

          <div style={{ marginTop: "auto", paddingTop: "16px" }}>
            {qualityResult.passed ? (
              <button style={styles.btnPrimary} onClick={() => setScreen("results")}>
                Continue to Analysis →
              </button>
            ) : (
              <button style={styles.btnPrimary} onClick={resetRecording}>
                🎙  Re-record
              </button>
            )}
            <button style={styles.btnOutline} onClick={resetRecording}>
              {qualityResult.passed ? "Re-record instead" : "← Back"}
            </button>
          </div>
        </>
      ) : null}
    </div>
  );

  // RESULTS (placeholder — model-dependent)
  const ResultsScreen = () => {
    const mockFeatures = [
      { name: "Fundamental Frequency (F0)", value: "182.4 Hz", status: "normal", desc: "Within expected range for reported demographics" },
      { name: "Harmonic-to-Noise Ratio", value: "18.7 dB", status: "flag", desc: "Below typical threshold — may indicate incomplete glottal closure" },
      { name: "Jitter (local)", value: "1.42%", status: "normal", desc: "Within expected range" },
      { name: "Shimmer (local)", value: "4.81%", status: "flag", desc: "Elevated — may indicate amplitude instability" },
    ];

    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <div style={styles.logo}>V</div>
          <div>
            <div style={styles.brandName}>VoiceSignal</div>
            <div style={styles.tagline}>Analysis Results</div>
          </div>
        </div>

        <div style={{ ...styles.card, background: "rgba(2, 128, 144, 0.06)", borderColor: "#0a4a50", textAlign: "center", padding: "28px" }}>
          <div style={{ fontSize: "11px", fontWeight: 600, color: "#4a7a6a", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "8px" }}>Voice Health Profile</div>
          <div style={{ fontSize: "48px", fontWeight: 700, color: "#f59e0b", fontFamily: "'DM Mono', monospace" }}>
            2 / 4
          </div>
          <div style={{ fontSize: "13px", color: "#6a9a8a", marginTop: "4px" }}>features flagged for review</div>
        </div>

        <div style={styles.card}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: "#02C39A", marginBottom: "16px", textTransform: "uppercase", letterSpacing: "1px" }}>
            Extracted Features
          </div>
          {mockFeatures.map((f, i) => (
            <div key={i} style={{ padding: "14px 0", borderBottom: i < mockFeatures.length - 1 ? "1px solid #1a3040" : "none" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "14px", fontWeight: 500, color: "#e0ede8" }}>{f.name}</span>
                <span style={{
                  ...styles.badge,
                  background: f.status === "normal" ? "rgba(2,195,154,0.12)" : "rgba(245,158,11,0.12)",
                  color: f.status === "normal" ? "#02C39A" : "#f59e0b",
                }}>
                  {f.status === "normal" ? "normal" : "review"}
                </span>
              </div>
              <div style={{ fontSize: "20px", fontWeight: 700, color: f.status === "normal" ? "#02C39A" : "#f59e0b", fontFamily: "'DM Mono', monospace", margin: "6px 0 4px" }}>
                {f.value}
              </div>
              <div style={{ fontSize: "12px", color: "#6a9a8a", lineHeight: 1.5 }}>{f.desc}</div>
            </div>
          ))}
        </div>

        <div style={{ ...styles.card, background: "rgba(245,158,11,0.06)", borderColor: "#3a2a10" }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: "#f59e0b", marginBottom: "8px" }}>What does this mean?</div>
          <div style={{ fontSize: "13px", color: "#c0b890", lineHeight: 1.7 }}>
            Some of your voice features fall outside typical ranges when compared to the Bridge2AI dataset.
            This is not a diagnosis — many factors affect voice quality including fatigue, hydration, and recording conditions.
            If you have persistent voice changes, consider discussing these results with an ENT specialist.
          </div>
        </div>

        <div style={{ ...styles.card, display: "none" /* placeholder for nearest-neighbor matches */ }}>
          <div style={{ fontSize: "13px", fontWeight: 600, color: "#02C39A", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "1px" }}>
            Similar Profiles in B2AI Dataset
          </div>
        </div>

        <div style={{ marginTop: "auto", paddingTop: "16px" }}>
          <button style={styles.btnPrimary} onClick={() => alert("PDF generation — will be implemented with backend")}>
            Download Report (PDF)
          </button>
          <button style={styles.btnOutline} onClick={() => { setScreen("landing"); setAudioBlob(null); setQualityResult(null); }}>
            Start Over
          </button>
        </div>

        <div style={styles.footer}>
          Results based on Bridge2AI Voice v3.0 (833 participants)<br />
          This tool is for informational purposes only — not a medical device.
        </div>
      </div>
    );
  };

  return (
    <div style={styles.app}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { margin: 0; background: #040e12; }
        button:hover { opacity: 0.9; }
        button:active { transform: scale(0.98); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      `}</style>
      {screen === "landing" && <LandingScreen />}
      {screen === "record" && <RecordScreen />}
      {screen === "quality" && <QualityScreen />}
      {screen === "results" && <ResultsScreen />}
    </div>
  );
}
