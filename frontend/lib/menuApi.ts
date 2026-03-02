/**
 * Menu API client and types for QuickBiteSB backend.
 */

export interface MenuItem {
  name: string;
  station?: string;
}

export interface Meal {
  name: string;
  stations: string[];
  items: MenuItem[];
}

export interface MenuResponse {
  hall: string;
  date: string;
  meals: Meal[];
}

const BACKEND_URL =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
    ? process.env.NEXT_PUBLIC_BACKEND_URL
    : "http://localhost:8000";

/** Map frontend display names (DiningCommons) to UCSB API hall slugs */
export const DINING_COMMONS_TO_SLUG: Record<string, string> = {
  "De la Guerra": "de-la-guerra",
  Carillo: "carrillo",
  Portola: "portola",
  Ortega: "ortega",
};

export function commonsToSlug(commons: string): string {
  return DINING_COMMONS_TO_SLUG[commons] ?? commons.toLowerCase().replace(/\s+/g, "-");
}

export async function fetchMenu(
  commons: string,
  date?: string
): Promise<MenuResponse> {
  const slug = commonsToSlug(commons);
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  const query = params.toString();
  const url = `${BACKEND_URL}/v1/menus/${encodeURIComponent(slug)}${query ? `?${query}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    const message =
      typeof detail.detail === "string" ? detail.detail : `Request failed: ${res.status}`;
    throw new Error(message);
  }
  return res.json() as Promise<MenuResponse>;
}
