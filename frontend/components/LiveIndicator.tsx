"use client";

type LiveIndicatorSize = "sm" | "md" | "lg";

interface LiveIndicatorProps {
  label?: string;
  color?: string;
  size?: LiveIndicatorSize;
  pulse?: boolean;
  showLabel?: boolean;
}

const sizes: Record<
  LiveIndicatorSize,
  { dot: number; ring: number; text: string; gap: number }
> = {
  sm: { dot: 8, ring: 8, text: "11px", gap: 6 },
  md: { dot: 10, ring: 10, text: "13px", gap: 8 },
  lg: { dot: 14, ring: 14, text: "16px", gap: 10 },
};

export function LiveIndicator({
  label = "LIVE",
  color = "#ff2d2d",
  size = "md",
  pulse = true,
  showLabel = true,
}: LiveIndicatorProps) {
  const s = sizes[size] ?? sizes.md;

  return (
    <div
      className="inline-flex items-center shrink-0"
      style={{ gap: s.gap }}
    >
      {/* Dot + pulse wrapper */}
      <div
        className="relative shrink-0"
        style={{ width: s.dot, height: s.dot }}
      >
        {pulse && (
          <span
            className="live-indicator-pulse absolute inset-0 rounded-full"
            style={{
              background: color,
              opacity: 0.4,
            }}
          />
        )}
        <span
          className="absolute inset-0 rounded-full"
          style={{
            background: color,
          }}
        />
      </div>
      {showLabel && (
        <span
          className="font-mono font-bold select-none"
          style={{
            fontSize: s.text,
            letterSpacing: "0.12em",
            color,
          }}
        >
          {label}
        </span>
      )}
    </div>
  );
}
