import type { DiningCommons } from "@/components/DiningCommonsSelector";

interface WaterTankProps {
  level: number;
  selectedCommons?: DiningCommons;
}

export function WaterTank({ level, selectedCommons }: WaterTankProps) {
  const clampedLevel = Number.isFinite(level)
    ? Math.min(1, Math.max(0, level))
    : 0;
  const percentage = Math.round(clampedLevel * 100);

  return (
    <div className="water-tank aspect-square w-full max-w-sm">
      <div
        className="water-fill"
        style={{ height: `${percentage}%` }}
        aria-hidden="true"
      >
        <div className="water-wave" />
      </div>

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
          <span>Updates periodically</span>
        </div>
      </div>
    </div>
  );
}

