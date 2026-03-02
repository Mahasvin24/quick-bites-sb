"use client";

import { useEffect, useMemo, useState } from "react";

import { cn } from "@/lib/utils";
import type { Meal, MenuResponse } from "@/lib/menuApi";

function getDefaultMealIndex(meals: Meal[]): number {
  if (!meals.length) return 0;
  const hour = new Date().getHours();
  const nameLower = (name: string) => (name || "").toLowerCase();
  const index = meals.findIndex((m) => {
    const n = nameLower(m.name);
    if (n.includes("breakfast") && hour >= 6 && hour < 11) return true;
    if (n.includes("lunch") && hour >= 11 && hour < 15) return true;
    if (n.includes("dinner") || n.includes("supper")) {
      if (hour >= 16 || hour < 9) return true;
    }
    if (n.includes("late") && hour >= 20) return true;
    if (n.includes("brunch") && hour >= 9 && hour < 14) return true;
    return false;
  });
  return index >= 0 ? index : 0;
}

interface MenuPanelProps {
  title?: string;
  subtitle?: string;
  menu: MenuResponse | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export function MenuPanel({
  title = "Today's Menu",
  subtitle = "Snapshot of featured items",
  menu,
  loading = false,
  error = null,
  onRetry,
}: MenuPanelProps) {
  const meals = menu?.meals ?? [];
  const [mealIndex, setMealIndex] = useState(0);

  useEffect(() => {
    setMealIndex(getDefaultMealIndex(meals));
  }, [meals]);

  const effectiveIndex = mealIndex >= meals.length ? 0 : mealIndex;
  const currentMeal = meals[effectiveIndex] ?? null;
  const mealOptions = useMemo(() => meals.map((m) => m.name), [meals]);
  const selectedMealName = currentMeal?.name ?? "";

  if (loading) {
    return (
      <section className="flex h-full flex-col rounded-3xl border border-border bg-card/80 p-5 shadow-sm sm:p-6 lg:p-7">
        <header className="mb-4 flex flex-col gap-1 sm:mb-6">
          <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
            {title}
          </h2>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </header>
        <div className="flex-1 space-y-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className="h-6 w-full max-w-[85%] animate-pulse rounded-lg bg-muted"
              style={{ width: `${70 + (i % 3) * 10}%` }}
            />
          ))}
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="flex h-full flex-col rounded-3xl border border-border bg-card/80 p-5 shadow-sm sm:p-6 lg:p-7">
        <header className="mb-4 flex flex-col gap-1 sm:mb-6">
          <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
            {title}
          </h2>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </header>
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
          <p className="text-sm text-muted-foreground">{error}</p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className={cn(
                "rounded-lg border border-border bg-muted px-4 py-2 text-sm font-medium",
                "hover:bg-muted/80 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              )}
            >
              Retry
            </button>
          )}
        </div>
      </section>
    );
  }

  const hasMealSelector = mealOptions.length > 1;

  return (
    <section className="flex h-full flex-col rounded-3xl border border-border bg-card/80 p-5 shadow-sm sm:p-6 lg:p-7">
      <header className="mb-4 flex flex-col gap-2 sm:mb-6">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold tracking-tight sm:text-xl">
            {title}
          </h2>
          {hasMealSelector && (
            <select
              aria-label="Select meal period"
              value={selectedMealName}
              onChange={(e) => {
                const i = mealOptions.indexOf(e.target.value);
                setMealIndex(i >= 0 ? i : 0);
              }}
              className={cn(
                "rounded-lg border border-input bg-background px-3 py-1.5 text-sm font-medium",
                "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              )}
            >
              {mealOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </header>

      <div className="relative -mx-1 flex-1 overflow-hidden">
        <div className="h-full overflow-auto px-1">
          {!currentMeal || !currentMeal.items.length ? (
            <p className="text-sm text-muted-foreground">
              No menu items for this period.
            </p>
          ) : (
            <div className="space-y-6">
              {(() => {
                const byStation = new Map<string, { name: string; items: string[] }>();
                for (const it of currentMeal.items) {
                  const station = it.station ?? "General";
                  if (!byStation.has(station)) {
                    byStation.set(station, { name: station, items: [] });
                  }
                  byStation.get(station)!.items.push(it.name);
                }
                return Array.from(byStation.values()).map((station) => (
                  <div key={station.name} className="space-y-2">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {station.name}
                    </h3>
                    <ul className="space-y-1">
                      {station.items.map((itemName) => (
                        <li
                          key={`${station.name}-${itemName}`}
                          className="flex items-center rounded-lg px-2 py-1.5 text-sm text-foreground/90 transition-colors hover:bg-muted/80"
                        >
                          <span>{itemName}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ));
              })()}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
