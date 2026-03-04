"use client";

import { useEffect, useState } from "react";

import {
  DiningCommonsSelector,
  type DiningCommons,
} from "@/components/DiningCommonsSelector";
import { MenuPanel } from "@/components/MenuPanel";
import { WaterTank } from "@/components/WaterTank";
import {
  fetchMenu,
  type MenuResponse,
  fetchOccupancyForCommons,
  type OccupancySnapshot,
} from "@/lib/menuApi";

export default function Home() {
  const [selectedCommons, setSelectedCommons] =
    useState<DiningCommons>("De la Guerra");
  const [menu, setMenu] = useState<MenuResponse | null>(null);
  const [menuError, setMenuError] = useState<string | null>(null);
  const [menuLoading, setMenuLoading] = useState(true);
  const [occupancy, setOccupancy] = useState<OccupancySnapshot | null>(null);
  const [occupancyError, setOccupancyError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setMenuLoading(true);
    setMenuError(null);
    fetchMenu(selectedCommons)
      .then((data) => {
        if (!cancelled) {
          setMenu(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setMenuError(err instanceof Error ? err.message : String(err));
          setMenu(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setMenuLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCommons]);

  useEffect(() => {
    let cancelled = false;
    setOccupancyError(null);
    setOccupancy(null);
    fetchOccupancyForCommons(selectedCommons)
      .then((data) => {
        if (!cancelled) {
          setOccupancy(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          // For now we just record the error; the UI will show 0% if null.
          setOccupancyError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCommons]);

  const waterLevel =
    occupancy && Number.isFinite(occupancy.percent_full)
      ? Math.max(0, Math.min(1, occupancy.percent_full / 100))
      : 0;

  return (
    <div className="flex h-screen w-screen flex-col bg-linear-to-b from-background to-muted text-foreground">
      <div className="mx-auto flex h-full w-full max-w-7xl flex-col px-5 py-5 sm:px-8 sm:py-8">
        <header className="mb-5 flex flex-col gap-3 sm:mb-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              QuickBiteSB
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Live feel for the dining commons at a glance.
            </p>
          </div>
          <a
            href="https://nutrition.info.dining.ucsb.edu/NetNutrition/1#"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            Nutritional Information
          </a>
        </header>

        <div className="mb-6 sm:mb-8">
          <DiningCommonsSelector
            value={selectedCommons}
            onChange={setSelectedCommons}
          />
        </div>

        <main className="flex min-h-0 flex-1 flex-col gap-6 sm:flex-row sm:gap-8">
          <section className="flex min-w-0 flex-1 shrink-0 items-center justify-center px-4 py-6 sm:flex-[5] sm:px-6 sm:py-8">
            <WaterTank level={waterLevel} selectedCommons={selectedCommons} />
          </section>

          <section className="min-w-0 flex-1 sm:flex-[9]">
            <MenuPanel
              subtitle={`${selectedCommons} · Featured selection`}
              menu={menu}
              loading={menuLoading}
              error={menuError}
              onRetry={() => {
                setMenuError(null);
                setMenuLoading(true);
                fetchMenu(selectedCommons)
                  .then(setMenu)
                  .catch((err) =>
                    setMenuError(err instanceof Error ? err.message : String(err))
                  )
                  .finally(() => setMenuLoading(false));
              }}
            />
          </section>
        </main>
      </div>
    </div>
  );
}
