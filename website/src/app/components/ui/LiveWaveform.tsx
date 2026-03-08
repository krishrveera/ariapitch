import { useRef, useEffect } from "react";

interface LiveWaveformProps {
    analyser: AnalyserNode | null;
    isActive: boolean;
}

export function LiveWaveform({ analyser, isActive }: LiveWaveformProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>(0);

    useEffect(() => {
        if (!analyser || !isActive || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d")!;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            animationRef.current = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const w = rect.width;
            const h = rect.height;

            ctx.clearRect(0, 0, w, h);

            // Draw waveform
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = "#7c3aed"; // violet-600
            ctx.beginPath();

            const sliceWidth = w / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * h) / 2;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.lineTo(w, h / 2);
            ctx.stroke();

            // Draw glow effect
            ctx.lineWidth = 6;
            ctx.strokeStyle = "rgba(124, 58, 237, 0.15)";
            ctx.beginPath();
            x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * h) / 2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }
            ctx.lineTo(w, h / 2);
            ctx.stroke();

            // Draw amplitude bars at the bottom
            analyser.getByteFrequencyData(dataArray);
            const barCount = 48;
            const barWidth = w / barCount - 2;
            const step = Math.floor(bufferLength / barCount);

            for (let i = 0; i < barCount; i++) {
                const value = dataArray[i * step];
                const barHeight = (value / 255) * h * 0.4;
                const alpha = 0.3 + (value / 255) * 0.7;

                ctx.fillStyle = `rgba(124, 58, 237, ${alpha})`;
                ctx.fillRect(
                    i * (barWidth + 2),
                    h - barHeight,
                    barWidth,
                    barHeight
                );
            }
        };

        draw();

        return () => {
            cancelAnimationFrame(animationRef.current);
        };
    }, [analyser, isActive]);

    return (
        <canvas
            ref={canvasRef}
            className="w-full h-24 sm:h-32 rounded-2xl"
            style={{ imageRendering: "auto" }}
        />
    );
}
