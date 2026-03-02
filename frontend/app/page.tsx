"use client";

import { useState } from "react";

import {
  DiningCommonsSelector,
  type DiningCommons,
} from "@/components/DiningCommonsSelector";
import { MenuPanel } from "@/components/MenuPanel";
import { WaterTank } from "@/components/WaterTank";

export default function Home() {
  const [selectedCommons, setSelectedCommons] =
    useState<DiningCommons>("De La Guerra");

  // Internal water level value between 0 and 1.
  const waterLevel = 0.72;

  return (
    <div className="flex h-screen w-screen flex-col bg-linear-to-b from-background to-muted text-foreground">
      <div className="mx-auto flex h-full w-full max-w-6xl flex-col px-5 py-5 sm:px-8 sm:py-8">
        <header className="mb-5 flex flex-col gap-3 sm:mb-8 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              QuickBiteSB
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Live feel for the dining commons at a glance.
            </p>
          </div>
        </header>

        <div className="mb-6 sm:mb-8">
          <DiningCommonsSelector
            value={selectedCommons}
            onChange={setSelectedCommons}
          />
        </div>

        <main className="flex min-h-0 flex-1 flex-col gap-6 sm:flex-row sm:gap-8">
          <section className="flex flex-1 items-center justify-center">
            <WaterTank level={waterLevel} selectedCommons={selectedCommons} />
          </section>

          <section className="flex flex-1 sm:flex-[1.2]">
            <MenuPanel
              subtitle={`${selectedCommons} · Featured selection`}
            />
          </section>
        </main>
      </div>
    </div>
  );
}
