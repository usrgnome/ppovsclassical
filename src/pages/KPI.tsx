// ===== KPI UI (drop-in) =====================================

import { HistoryEntry } from "../types/types";
import React, { useEffect, useMemo, useState } from "react";

type KPISectionProps = {
  day: number;
  episodeProfit: number;
  history: HistoryEntry[];
  footTraffic: number[];
  sales: number;
  holdingCost: number;
  restockCost: number;
  profit: number; // profit after costs (today)
  currentPrice: number; // today’s price
};

function pct(n: number) {
  if (!isFinite(n) || isNaN(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}
function money(n: number) {
  if (!isFinite(n) || isNaN(n)) return "—";
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}
function money2(n: number) {
  if (!isFinite(n) || isNaN(n)) return "—";
  return `$${n.toFixed(2)}`;
}
function arrWindow<T>(arr: T[], endIdx: number, len = 7) {
  const start = Math.max(0, endIdx - len + 1);
  return arr.slice(start, endIdx + 1);
}

function tinySpark(points: number[]) {
  const max = Math.max(1, ...points);
  const min = Math.min(...points);
  // normalize 0..100
  const norm = points.map((p) =>
    max === min ? 50 : ((p - min) / (max - min)) * 100
  );
  const step = points.length > 1 ? 100 / (points.length - 1) : 100;
  const d = norm.map((n, i) => `${i * step},${100 - n}`).join(" ");
  return (
    <svg viewBox="0 0 100 100" className="h-6 w-20 opacity-80">
      <polyline fill="none" stroke="currentColor" strokeWidth="6" points={d} />
    </svg>
  );
}

function Delta({ value, suffix = "" }: { value: number; suffix?: string }) {
  if (!isFinite(value)) return null;
  const up = value > 0;
  const flat = value === 0;
  return (
    <span
      className={
        "ml-2 inline-flex items-center rounded px-1.5 py-0.5 text-[10px] text-white m-1 " +
        (flat
          ? "bg-white/10"
          : up
          ? "bg-green-500 text-green-100"
          : "bg-red-500 text-red-100")
      }
      aria-label={`change ${up ? "up" : flat ? "flat" : "down"} ${Math.abs(
        value
      ).toFixed(1)}${suffix}`}
    >
      {!flat && (up ? "▲" : "▼")} {Math.abs(value).toFixed(1)}
      {suffix}
    </span>
  );
}

function MetricCardKPI({
  title,
  value,
  help,
  spark,
  delta,
  // formatter now expects a NUMBER and returns a formatted string
  formatter,
}: {
  title: string;
  value: string | number;
  help: string;
  spark?: number[];
  delta?: { absolute?: number; percent?: number };
  formatter?: (n: number) => string;
}) {
  // Decide what to show
  let shown: string;
  if (typeof value === "number") {
    shown = formatter ? formatter(value) : value.toLocaleString();
  } else {
    // If you already passed a preformatted string (e.g., "12.5%"), just show it
    shown = value;
  }

  return (
    <div className="bg-[#7d4321] text-[#ffe5c1] col-span-12 rounded-2xl border-4 border-[#7d4321] p-4 shadow-[0_8px_0_#693619,0_14px_24px_rgba(0,0,0,0.25)] sm:col-span-6 lg:col-span-4">
      <div className="flex items-center shadowed justify-between">
        <h3 className="text-[11px] tracking-wider">{title}</h3>
      </div>
      <div
        className="leading-none text-4xl font-semibold inline-flex"
        aria-live="polite"
      >
        <div className="shadowed">{shown}</div>
        {delta?.percent !== undefined && (
          <Delta value={delta.percent} suffix="%" />
        )}
        {delta?.absolute !== undefined && <Delta value={delta.absolute} />}
      </div>
      <div className="font-rowdies shadowed text-xs opacity-90">{help}</div>
    </div>
  );
}

export function KPISection({
  day,
  history,
  footTraffic,
  episodeProfit,
  sales,
  holdingCost,
  restockCost,
  profit,
  currentPrice,
}: KPISectionProps) {
  // Pull time series from history (make safe fallbacks)
  const salesSeries = history.map((h) => h.sales ?? 0);
  const footSeries = footTraffic.map((f) => f ?? 0);
  const revenueSeries = history.map(
    (h) => h.revenue ?? (h.sales ?? 0) * (h.price ?? currentPrice)
  );
  const wasteSeries = history.map((h) => h.expired ?? 0);

  const todaySales = sales ?? salesSeries[day] ?? 0;
  const todayFoot = footTraffic[day] ?? footSeries[day] ?? 0;
  const todayRevenue =
    revenueSeries[day] ?? todaySales * (history[day]?.price ?? currentPrice);

  // KPIs
  const conversion = todayFoot ? todaySales / todayFoot : 0;
  const aov = todaySales ? todayRevenue / todaySales : currentPrice;

  const opening = history[day]?.openingStock ?? 0;
  const restocked = history[day]?.restocked ?? 0;
  const sellThrough = todaySales / Math.max(1, opening + restocked) || 0;

  const waste = history[day]?.expired ?? 0;

  // Deltas vs yesterday & 7-day avg (percent)
  const yIdx = Math.max(0, day - 1);
  const prevSales = salesSeries[yIdx] ?? 0;
  const prevConv =
    (footSeries[yIdx] ? (salesSeries[yIdx] || 0) / footSeries[yIdx] : 0) || 0;
  const prevAOV = salesSeries[yIdx]
    ? revenueSeries[yIdx] / salesSeries[yIdx]
    : currentPrice;

  const last7Sales = arrWindow(salesSeries, day, 7);
  const avg7Sales =
    last7Sales.reduce((a, b) => a + b, 0) / Math.max(1, last7Sales.length);

  const deltaPct = (now: number, prev: number) =>
    prev === 0 ? (now > 0 ? 100 : 0) : ((now - prev) / prev) * 100;

  const convDeltaPct = deltaPct(conversion, prevConv);
  const aovDeltaPct = deltaPct(aov, prevAOV);
  const salesDeltaPct = deltaPct(todaySales, prevSales);
  const salesVs7Pct = deltaPct(todaySales, avg7Sales);

  return (
    <div className="grid grid-cols-12 gap-4">
      {/* Traffic */}
      <MetricCardKPI
        title="FOOT TRAFFIC"
        value={todayFoot}
        help="PEOPLE CAME IN"
        spark={arrWindow(footSeries, day, 7)}
        delta={{ absolute: todayFoot - (footSeries[yIdx] ?? 0) }}
      />
      {/* Sales */}
      <MetricCardKPI
        title="SALES"
        value={todaySales}
        help="ITEMS SOLD"
        spark={arrWindow(salesSeries, day, 7)}
        delta={{ absolute: todaySales - prevSales }}
      />
      {/* Profit */}
      <MetricCardKPI
        title="TODAYS PROFIT"
        value={profit}
        help="AFTER COSTS FOR TODAY"
        spark={arrWindow(
          revenueSeries.map((r, i) => r - (history[i]?.cost ?? 0)),
          day,
          7
        )}
        formatter={money}
        delta={{ absolute: profit - (history[yIdx]?.profit ?? 0) }}
      />

      <MetricCardKPI
        title="EPISODE PROFIT"
        value={episodeProfit}
        help="AFTER COSTS FOR EPISODE"
        formatter={money}
        delta={{
          absolute: episodeProfit - (history[yIdx]?.episodeProfit ?? 0),
        }}
      />

      {/* New: Conversion */}
      <MetricCardKPI
        title="CONVERSION"
        value={pct(conversion)}
        help="VISITORS → BUYERS"
        spark={arrWindow(
          history.map((_, i) =>
            footSeries[i] ? (salesSeries[i] || 0) / footSeries[i] : 0
          ),
          day,
          7
        )}
        // show percent delta vs yesterday
        delta={{ percent: convDeltaPct }}
      />

      {/* New: Sell-through */}
      <MetricCardKPI
        title="SELL-THROUGH"
        value={pct(sellThrough)}
        help="STOCK MOVED"
        spark={arrWindow(
          history.map((h, i) => {
            const os = h.openingStock ?? 0;
            const rs = h.restocked ?? 0;
            const s = salesSeries[i] ?? 0;
            return os + rs ? s / (os + rs) : 0;
          }),
          day,
          7
        )}
        delta={{
          percent: deltaPct(
            sellThrough,
            (() => {
              const os = history[yIdx]?.openingStock ?? 0;
              const rs = history[yIdx]?.restocked ?? 0;
              const s = salesSeries[yIdx] ?? 0;
              return os + rs ? s / (os + rs) : 0;
            })()
          ),
        }}
      />

      {/* New: Waste / Spoilage (if tracked) */}
      <MetricCardKPI
        title="WASTE"
        value={waste}
        help="EXPIRED TODAY"
        spark={arrWindow(wasteSeries, day, 7)}
        delta={{ absolute: waste - (wasteSeries[yIdx] ?? 0) }}
      />

      {/* New: Waste / Spoilage (if tracked) */}
      <MetricCardKPI
        title="HOLDING COST"
        value={holdingCost}
        help="COST OF HOLDING INVENTORY"
        delta={{ absolute: holdingCost - (history[yIdx]?.holding_cost ?? 0) }}
      />

      {/* New: Waste / Spoilage (if tracked) */}
      <MetricCardKPI
        title="RESTOCK COST"
        value={restockCost}
        help="COST OF RESTOCKING INVENTORY"
        delta={{ absolute: restockCost - (history[yIdx]?.restock_cost ?? 0) }}
      />
    </div>
  );
}
// ===== end KPI UI ============================================
