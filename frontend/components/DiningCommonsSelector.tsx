import { cn } from "@/lib/utils";

const DINING_COMMONS = ["De La Guerra", "Carillo", "Portola", "Ortega"] as const;

export type DiningCommons = (typeof DINING_COMMONS)[number];

interface DiningCommonsSelectorProps {
  value: DiningCommons;
  onChange: (value: DiningCommons) => void;
  className?: string;
}

export function DiningCommonsSelector({
  value,
  onChange,
  className,
}: DiningCommonsSelectorProps) {
  return (
    <div
      className={cn(
        "inline-flex flex-wrap gap-2 rounded-full bg-muted/60 p-1.5 text-sm",
        className,
      )}
    >
      {DINING_COMMONS.map((option) => {
        const isActive = option === value;

        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={cn(
              "flex items-center justify-center rounded-full px-4 py-1.5 transition-colors",
              "outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              isActive
                ? "bg-primary text-primary-foreground shadow-sm"
                : "bg-transparent text-muted-foreground hover:bg-background/80",
            )}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}

