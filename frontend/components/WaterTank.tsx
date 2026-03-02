import type { DiningCommons } from "@/components/DiningCommonsSelector";
import { cn } from "@/lib/utils";
import Wave from "react-wavify";

interface WaterTankProps {
  level: number;
  selectedCommons?: DiningCommons;
}

export function WaterTank({ level, selectedCommons }: WaterTankProps) {
  const clampedLevel = Number.isFinite(level)
    ? Math.min(1, Math.max(0, level))
    : 0;
  const percentage = Math.round(clampedLevel * 100);
  const isCarillo = selectedCommons === "Carillo";

  return (
    <div
      className={cn(
        "water-tank aspect-square w-full max-w-sm",
        isCarillo && "water-tank--carillo max-w-md",
      )}
    >
      <div
        className={cn("water-fill", isCarillo && "water-fill--carillo")}
        style={{ height: `${clampedLevel * 100}%` }}
        aria-hidden="true"
      >
        <Wave
          fill="oklch(0.76 0.11 215)"
          paused={false}
          options={{
            height: 18,
            amplitude: 10 + clampedLevel * 18,
            speed: 0.18,
            points: 3,
          }}
          style={{
            width: "200%",
            height: "120%",
            transform: "translateX(-25%)",
          }}
        />
      </div>

      {isCarillo && (
        <div
          aria-hidden="true"
          className="water-carillo-outline pointer-events-none absolute inset-6 sm:inset-7"
        />
      )}

      <div className="water-tank-inner">
        <div className="flex items-center justify-between text-[0.7rem] font-medium uppercase tracking-[0.2em] text-muted-foreground">
          <span>Occupancy</span>
          <span>{percentage}% full</span>
        </div>

        <div className="flex flex-1 items-center justify-center">
          <div className="text-center">
            <div className="text-4xl font-semibold tracking-tight sm:text-5xl">
              {percentage}
              <span className="text-base font-normal text-muted-foreground">
                %
              </span>
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              {selectedCommons ?? "Dining Commons"}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-[0.7rem] text-muted-foreground">
          <span>Live estimate</span>
          <span>Animated surface</span>
        </div>
      </div>
    </div>
  );
}

