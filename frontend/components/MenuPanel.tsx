"use client";

import { useEffect, useMemo, useState } from "react";

import { cn } from "@/lib/utils";
import type { Meal, MenuResponse } from "@/lib/menuApi";

/** Meal windows: Mon–Fri Breakfast 7:15–10, Lunch 11–3, Dinner 5–8:30; Sat/Sun Brunch 10–2, Dinner 5–8:30 */
function getDefaultMealIndex(meals: Meal[]): number {
  if (!meals.length) return 0;
  const now = new Date();
  const day = now.getDay(); // 0 = Sunday, 6 = Saturday
  const minSinceMidnight = now.getHours() * 60 + now.getMinutes();
  const isWeekend = day === 0 || day === 6;

  let desiredName: string;
  if (isWeekend) {
    // Brunch 10:00–14:00, Dinner 17:00–20:30
    if (minSinceMidnight >= 10 * 60 && minSinceMidnight < 14 * 60) desiredName = "brunch";
    else if (minSinceMidnight >= 17 * 60 && minSinceMidnight < 20 * 60 + 30) desiredName = "dinner";
    else desiredName = minSinceMidnight < 14 * 60 ? "brunch" : "dinner";
  } else {
    // Weekday: Breakfast 7:15–10:00, Lunch 11:00–15:00, Dinner 17:00–20:30
    if (minSinceMidnight >= 7 * 60 + 15 && minSinceMidnight < 10 * 60) desiredName = "breakfast";
    else if (minSinceMidnight >= 11 * 60 && minSinceMidnight < 15 * 60) desiredName = "lunch";
    else if (minSinceMidnight >= 17 * 60 && minSinceMidnight < 20 * 60 + 30) desiredName = "dinner";
    else desiredName = minSinceMidnight < 10 * 60 ? "breakfast" : minSinceMidnight < 15 * 60 ? "lunch" : "dinner";
  }

  const nameLower = (name: string) => (name || "").toLowerCase();
  const index = meals.findIndex((m) => nameLower(m.name).includes(desiredName));
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
        <div className="grid flex-1 grid-cols-2 gap-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className="h-6 animate-pulse rounded-lg bg-muted"
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
            <div className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2">
              {(() => {
                const byStation = new Map<
                  string,
                  { name: string; items: string[] }
                >();
                for (const it of currentMeal.items) {
                  const station = it.station ?? "General";
                  if (!byStation.has(station)) {
                    byStation.set(station, { name: station, items: [] });
                  }
                  byStation.get(station)!.items.push(it.name);
                }

                const stations = Array.from(byStation.values());
                const midpoint = Math.ceil(stations.length / 2);
                const leftColumn = stations.slice(0, midpoint);
                const rightColumn = stations.slice(midpoint);

                const renderColumn = (
                  columnStations: { name: string; items: string[] }[]
                ) => (
                  <div className="space-y-4">
                    {columnStations.map((station) => (
                      <div key={station.name} className="space-y-1">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          {station.name}
                        </h3>
                        <ul className="space-y-0.5">
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
                    ))}
                  </div>
                );

                return (
                  <>
                    {renderColumn(leftColumn)}
                    {rightColumn.length > 0 && renderColumn(rightColumn)}
                  </>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
