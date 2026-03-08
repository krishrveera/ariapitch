import { useRef, useEffect, useState } from "react";

interface RecordedWaveformProps {
    audioData: Float32Array;
    jitter?: number;   // percentage (e.g., 1.2)
    shimmer?: number;  // percentage (e.g., 3.4)
}

export function RecordedWaveform({ audioData, jitter = 1.2, shimmer = 3.4 }: RecordedWaveformProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>(0);
    const [phase, setPhase] = useState(0);

    useEffect(() => {
        if (!canvasRef.current || !audioData || audioData.length === 0) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d")!;

        // Generate jitter and shimmer marker positions (simulated)
        const totalSamples = audioData.length;
        const jitterPositions: number[] = [];
        const shimmerPositions: number[] = [];

        // Place markers at semi-random positions influenced by the actual values
        const jitterCount = Math.max(3, Math.round(jitter * 4));
        const shimmerCount = Math.max(3, Math.round(shimmer * 3));

        for (let i = 0; i < jitterCount; i++) {
            jitterPositions.push(
                Math.floor((i / jitterCount) * totalSamples * 0.8 + totalSamples * 0.1 + Math.random() * totalSamples * 0.05)
            );
        }
        for (let i = 0; i < shimmerCount; i++) {
            shimmerPositions.push(
                Math.floor((i / shimmerCount) * totalSamples * 0.8 + totalSamples * 0.1 + Math.random() * totalSamples * 0.05)
            );
        }

        let currentPhase = 0;

        const draw = () => {
            animationRef.current = requestAnimationFrame(draw);
            currentPhase += 0.03;

            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const w = rect.width;
            const h = rect.height;

            ctx.clearRect(0, 0, w, h);

            // Draw center line
            ctx.strokeStyle = "rgba(124, 58, 237, 0.08)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();

            // Downsample for drawing
            const step = Math.max(1, Math.floor(totalSamples / w));

            // Draw main waveform
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "#7c3aed"; // violet-600
            ctx.beginPath();

            for (let x = 0; x < w; x++) {
                const sampleIndex = Math.floor((x / w) * totalSamples);
                let min = 1.0, max = -1.0;
                for (let j = 0; j < step; j++) {
                    const idx = sampleIndex + j;
                    if (idx < totalSamples) {
                        const val = audioData[idx];
                        if (val < min) min = val;
                        if (val > max) max = val;
                    }
                }
                const yMin = ((1 + min) / 2) * h;
                const yMax = ((1 + max) / 2) * h;

                if (x === 0) {
                    ctx.moveTo(x, yMin);
                }
                ctx.lineTo(x, yMin);
                ctx.lineTo(x, yMax);
            }
            ctx.stroke();

            // Waveform fill
            ctx.fillStyle = "rgba(124, 58, 237, 0.08)";
            ctx.beginPath();
            for (let x = 0; x < w; x++) {
                const sampleIndex = Math.floor((x / w) * totalSamples);
                let max = -1.0;
                for (let j = 0; j < step; j++) {
                    const idx = sampleIndex + j;
                    if (idx < totalSamples) {
                        const val = audioData[idx];
                        if (val > max) max = val;
                    }
                }
                const yMax = ((1 + max) / 2) * h;
                if (x === 0) ctx.moveTo(x, h / 2);
                ctx.lineTo(x, yMax);
            }
            ctx.lineTo(w, h / 2);
            ctx.closePath();
            ctx.fill();

            // Draw JITTER markers (timing irregularities - red)
            for (const pos of jitterPositions) {
                const x = (pos / totalSamples) * w;
                const pulse = Math.sin(currentPhase * 2) * 0.5 + 0.5;
                const alpha = 0.3 + pulse * 0.5;
                const radius = 3 + pulse * 3;

                // Vertical highlight line
                ctx.strokeStyle = `rgba(244, 63, 94, ${alpha * 0.3})`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();

                // Dot marker
                const sampleIndex = Math.min(pos, totalSamples - 1);
                const val = audioData[sampleIndex];
                const y = ((1 + val) / 2) * h;

                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(244, 63, 94, ${alpha})`;
                ctx.fill();

                // Glow
                ctx.beginPath();
                ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(244, 63, 94, ${alpha * 0.2})`;
                ctx.fill();
            }

            // Draw SHIMMER markers (amplitude irregularities - amber)
            for (const pos of shimmerPositions) {
                const x = (pos / totalSamples) * w;
                const pulse = Math.sin(currentPhase * 1.5 + 1) * 0.5 + 0.5;
                const alpha = 0.3 + pulse * 0.5;
                const radius = 3 + pulse * 2;

                // Horizontal amplitude band
                const sampleIndex = Math.min(pos, totalSamples - 1);
                const val = audioData[sampleIndex];
                const y = ((1 + val) / 2) * h;

                ctx.fillStyle = `rgba(251, 191, 36, ${alpha * 0.15})`;
                ctx.fillRect(x - 8, y - 12, 16, 24);

                // Dot marker
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(251, 191, 36, ${alpha})`;
                ctx.fill();

                // Glow
                ctx.beginPath();
                ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(251, 191, 36, ${alpha * 0.2})`;
                ctx.fill();
            }
        };

        draw();

        return () => {
            cancelAnimationFrame(animationRef.current);
        };
    }, [audioData, jitter, shimmer]);

    return (
        <div className="space-y-2">
            <canvas
                ref={canvasRef}
                className="w-full h-28 sm:h-36 rounded-xl"
                style={{ imageRendering: "auto" }}
            />
            <div className="flex items-center gap-4 text-xs text-slate-500">
                <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                    Jitter ({jitter}%)
                </div>
                <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    Shimmer ({shimmer}%)
                </div>
            </div>
        </div>
    );
}
