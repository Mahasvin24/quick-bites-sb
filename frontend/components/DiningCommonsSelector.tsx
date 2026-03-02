"use client";

import { cn } from "@/lib/utils";
import { useLayoutEffect, useRef, useState } from "react";

const DINING_COMMONS = ["De la Guerra", "Carillo", "Portola", "Ortega"] as const;

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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const buttonRefs = useRef(new Map<DiningCommons, HTMLButtonElement>());
  const [indicator, setIndicator] = useState<{
    x: number;
    width: number;
    ready: boolean;
  }>({ x: 0, width: 0, ready: false });

  const updateIndicator = () => {
    const container = containerRef.current;
    const activeButton = buttonRefs.current.get(value);

    if (!container || !activeButton) return;

    const containerRect = container.getBoundingClientRect();
    const buttonRect = activeButton.getBoundingClientRect();

    setIndicator({
      x: buttonRect.left - containerRect.left,
      width: buttonRect.width,
      ready: true,
    });
  };

  useLayoutEffect(() => {
    updateIndicator();
  }, [value]);

  useLayoutEffect(() => {
    updateIndicator();

    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => updateIndicator());
    observer.observe(container);
    window.addEventListener("resize", updateIndicator);

    return () => {
      window.removeEventListener("resize", updateIndicator);
      observer.disconnect();
    };
  }, [value]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative inline-flex flex-nowrap items-center gap-1 rounded-full bg-muted/60 p-1.5 text-sm",
        className,
      )}
    >
      <div
        aria-hidden="true"
        className={cn(
          "pointer-events-none absolute inset-y-1.5 left-0 rounded-full bg-primary shadow-sm",
          "transition-[transform,width,opacity] duration-300 ease-out will-change-transform motion-reduce:transition-none",
          indicator.ready ? "opacity-100" : "opacity-0",
        )}
        style={{
          width: indicator.width,
          transform: `translateX(${indicator.x}px)`,
        }}
      />
      {DINING_COMMONS.map((option) => {
        const isActive = option === value;

        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            ref={(node) => {
              if (node) buttonRefs.current.set(option, node);
              else buttonRefs.current.delete(option);
            }}
            className={cn(
              "relative z-10 flex items-center justify-center rounded-full px-4 py-1.5 whitespace-nowrap transition-colors",
              "outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              isActive
                ? "text-primary-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
            aria-pressed={isActive}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}

