import React, { useEffect, useMemo, useState } from "react";
import { HistoryEntry } from "../types/types";
import DemandChart from "./DemandChart";
import RevenueChart from "./RevenueChart";
import { KPISection } from "./KPI";
import ConversionPriceChart from "./ConversionCharts";

type Mode = "day" | "night";

const COST_PER_UNIT = 4; // TODO: pass from server
const DEFAULT_PRICE = 10; // TODO: pass from server
const MAX_PRICE = 2 * DEFAULT_PRICE;

export interface Props {
  inventoryCapacity: number;
  done: boolean;
  episodeProfit: number;
  day: number;
  maxRestock: number;
  totalDays: number;
  currentPrice: number;
  inventoryLeft: number;
  shelfLife: number;
  profit: number;
  footTraffic: number[];
  sales: number;
  history: HistoryEntry[];
  batches: number[][];
  onConfirm: (newPrice: number, restockQty: number) => void;
  aiSuggestedPrice?: number;
  aiSuggestedRestockQty?: number;
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function useStarfield(count = 120) {
  return useMemo(
    () =>
      Array.from({ length: count }).map(() => ({
        left: Math.random() * 100,
        top: Math.random() * 60 + 10,
        delay: Math.random() * 3,
      })),
    [count]
  );
}

function AIRecommendationCard({
  suggestedPrice,
  suggestedQty,
  maxRestock,
  onUse,
  onUseAndStart,
  disabled,
}: {
  suggestedPrice?: number;
  suggestedQty?: number;
  maxRestock: number;
  onUse: () => void;
  onUseAndStart?: () => void;
  disabled?: boolean;
}) {
  const hasPrice = typeof suggestedPrice === "number";
  const hasQty = typeof suggestedQty === "number";
  const isDisabled = disabled || (!hasPrice && !hasQty);

  return (
    <div className="bg-[#683719] rounded-xl p-4 space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold">AI Recommended</h3>
      </div>
      <ul className="text-sm space-y-1">
        <li>
          Price:{" "}
          <b>{hasPrice ? `$${(suggestedPrice ?? 0).toFixed(2)}` : "—"}</b>
        </li>
        <li>
          Restock:{" "}
          <b>
            {hasQty
              ? `${Math.min(maxRestock, Math.max(0, suggestedQty!))} units`
              : "—"}
          </b>
        </li>
      </ul>
      <div className="flex gap-2 pt-1">
        <button
          disabled={isDisabled}
          onClick={onUse}
          className={`rounded-2xl border-4 border-[#3a322b] px-3 py-2 text-xs shadow-[0_6px_0_#4a154b] active:translate-y-0.5 active:shadow-[0_2px_0_#4a154b] ${
            isDisabled
              ? "bg-[#4b3a2f]/50 text-white/40 cursor-not-allowed"
              : "bg-[#dec036] text-[#4a154b]"
          }`}
          aria-disabled={isDisabled}
          aria-label="Use AI recommended price and restock"
        >
          Use
        </button>

        {onUseAndStart && (
          <button
            disabled={isDisabled}
            onClick={onUseAndStart}
            className={`rounded-2xl border-4 border-[#3a322b] px-3 py-2 text-xs shadow-[0_6px_0_#4a154b] active:translate-y-0.5 active:shadow-[0_2px_0_#4a154b] ${
              isDisabled
                ? "bg-[#4b3a2f]/50 text-white/40 cursor-not-allowed"
                : "bg-[#a8c637] text-[#2b2b2b]"
            }`}
            aria-disabled={isDisabled}
            aria-label="Use AI suggestion and start next day"
          >
            Use & Start ▶
          </button>
        )}
      </div>
    </div>
  );
}

function InventoryBatchVisualizer({
  inventoryLeft,
  currentDay,
  batches,
  shelfLife,
  inventoryCapacity,
}: {
  shelfLife: number;
  currentDay: number;
  inventoryLeft: number;
  batches: number[][];
  inventoryCapacity: number;
}) {
  return (
    <div className="bg-[#683719] p-4 rounded-xl shadow-md w-full">
      <h3 className="text-xl font-bold mb-3 shadowed text-[#ffe5c1]">
        Inventory Batches : {inventoryLeft} / {inventoryCapacity} units
      </h3>
      {batches.length === 0 ? (
        <p className="shadowed text-[#ffe5c1]">
          No inventory batches available.
        </p>
      ) : (
        <div className="overflow-y-auto max-h-[500px] pr-1">
          <ul className="space-y-3">
            {batches.map((batch, index) => {
              const [dayCreated, quantity] = batch;
              const expireDay = dayCreated + shelfLife;
              const daysLeft = expireDay - currentDay;
              const percentLeft = Math.max(0, (daysLeft / shelfLife) * 100);
              const color =
                percentLeft > 30
                  ? "bg-green-400"
                  : percentLeft > 10
                  ? "bg-yellow-400"
                  : "bg-red-500";
              return (
                <li
                  key={index}
                  className="border border-gray-200/20 rounded-md p-3 flex flex-col space-y-1"
                >
                  <div className="flex justify-between font-medium">
                    <span className="shadowed text-[#ffe5c1]">
                      Batch #{index + 1}
                    </span>
                    <span className="shadowed text-[#ffe5c1]">
                      {quantity} units
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-black/20 rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-3 rounded-full ${color}`}
                        style={{ width: `${percentLeft}%` }}
                      />
                    </div>
                    <span className="text-xs shadowed text-[#ffe5c1]">
                      {daysLeft > 0 ? `${daysLeft} day(s) left` : "Expired"}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function EpisodeOverOverlay({
  profit,
  onClose,
}: {
  profit: number;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      role="dialog"
      aria-modal="true"
      aria-label="Episode finished"
    >
      <div className="relative rounded-3xl border-8 border-[#3a322b] bg-[#7d4321] text-[#ffe5c1] p-8 text-center shadow-[0_12px_0_#3a322b,0_28px_40px_rgba(0,0,0,0.35)] max-w-md mx-auto">
        {/* Close (X) */}
        <button
          onClick={onClose}
          aria-label="Close"
          className="absolute right-3 top-3 rounded-full border-2 border-[#3a322b] bg-[#a85a2a] px-2 py-1 text-sm leading-none shadow-[0_2px_0_#3a322b] hover:brightness-110 focus:outline-none"
        >
          ✕
        </button>

        <div className="text-3xl font-bold mb-2">Episode Complete 🎉</div>
        <div className="text-lg mb-4">Final Profit</div>
        <div className="text-4xl font-extrabold mb-6">
          ${profit.toLocaleString()}
        </div>
        <div className="opacity-80 text-sm">
          You can review charts and history, or start a new run.
        </div>
      </div>
    </div>
  );
}

export default function RetroShopUI({
  day,
  done,
  maxRestock,
  currentPrice,
  inventoryLeft,
  inventoryCapacity,
  sales,
  history,
  batches,
  profit,
  episodeProfit,
  onConfirm,
  shelfLife,
  aiSuggestedPrice,
  aiSuggestedRestockQty,
  totalDays,
  footTraffic,
}: Props) {
  const [mode, setMode] = useState<Mode>("day");
  const [qty, setQty] = useState(0);
  const [pricePlan, setPricePlan] = useState(currentPrice || DEFAULT_PRICE);

  // NEW: allow dismissing the "done" overlay
  const [hideDoneOverlay, setHideDoneOverlay] = useState(false);
  useEffect(() => {
    // when a new episode ends, show the overlay again
    if (done) setHideDoneOverlay(false);
  }, [done]);

  const stars = useStarfield(120);

  // keep slider in range if server shrinks maxRestock
  useEffect(() => {
    setQty((q) => clamp(q, 0, maxRestock));
  }, [maxRestock]);

  // update planning price when server price changes
  useEffect(() => {
    setPricePlan(currentPrice || DEFAULT_PRICE);
  }, [currentPrice, mode]);

  const newInventoryAfterRestock = useMemo(
    () => Math.max(0, inventoryLeft + qty),
    [inventoryLeft, qty]
  );

  const goNight = () => setMode("night");
  const goDay = () => setMode("day");

  const nextDay = (price: number, restock: number) => {
    setMode("day");
    onConfirm(price, restock);
    setQty(0);
  };

  const applyAISuggestion = () => {
    if (typeof aiSuggestedPrice === "number") setPricePlan(aiSuggestedPrice);
    if (typeof aiSuggestedRestockQty === "number")
      setQty(clamp(aiSuggestedRestockQty, 0, maxRestock));
  };

  const useAndStart = () => {
    const price =
      typeof aiSuggestedPrice === "number" ? aiSuggestedPrice : pricePlan;
    const qtyClamped =
      typeof aiSuggestedRestockQty === "number"
        ? clamp(aiSuggestedRestockQty, 0, maxRestock)
        : qty;
    nextDay(price, qtyClamped);
  };

  return (
    <div
      className={
        (mode === "night"
          ? "bg-gradient-to-b from-[#3b0f52] via-[#2b0a3a] to-[#1b0824] text-[#ffeefc]"
          : "bg-gradient-to-b from-[#7e2de0] via-[#ff63c3] to-[#ffb86b] text-[#2a1330]") +
        " min-h-screen relative"
      }
    >
      <div className="pointer-events-none fixed inset-0 mix-blend-multiply opacity-40 [background:repeating-linear-gradient(0deg,rgba(255,255,255,0.07)_0px,rgba(255,255,255,0.07)_1px,transparent_2px,transparent_3px)]" />

      {done && !hideDoneOverlay && (
        <EpisodeOverOverlay
          profit={episodeProfit}
          onClose={() => setHideDoneOverlay(true)}
        />
      )}

      <div className="grid min-h-screen grid-rows-[auto,1fr,auto]">
        {/* Header */}
        <header className="relative flex items-center justify-between gap-3 p-4">
          <div className="flex items-center gap-3">
            <div
              className={
                (mode === "night"
                  ? "shadow-[0_0_22px_0_#9b6cff] bg-[radial-gradient(circle_at_40%_35%,#ffd6ff_0_35%,#b88cff_36%_100%)]"
                  : "shadow-[0_0_22px_0_#ffdd7a] bg-[radial-gradient(circle_at_35%_35%,#fff38a_0_35%,#ffcd38_36%_100%)]") +
                " relative h-12 w-12 rounded-full"
              }
            >
              <div className="absolute inset-0 opacity-50">
                <i className="absolute left-5 top-[9px] h-2.5 w-2.5 rounded-full bg-white/70" />
                <i className="absolute left-3.5 top-[22px] h-1.5 w-1.5 rounded-full bg-white/60" />
                <i className="absolute left-7 top-[30px] h-2 w-2 rounded-full bg-white/60" />
              </div>
            </div>
            <div className="text-2xl uppercase tracking-wide drop-shadow shadowed text-[#ffe5c1]">
              RETRO MART
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="px-6 py-3 rounded-lg border-4 border-[#3a322b] bg-[#cf7830] text-[#ffe5c1] text-sm shadow-[inset_0_-6px_0_#a55612] shadowed">
              Day {day + 1} / {totalDays}
            </div>
            <div
              className={`px-6 py-3 rounded-lg border-4 border-[#3a322b] text-sm shadowed ${
                mode === "night"
                  ? "bg-[#8354b5] text-[#ffe5c1] shadow-[inset_0_-6px_0_#604794]"
                  : "bg-[#a8c637] text-[#ffe5c1] shadow-[inset_0_-6px_0_#6b9930]"
              }`}
            >
              {mode === "night"
                ? "NIGHT — Plan & Restock"
                : "DAYTIME — Open for Business"}
            </div>
          </div>
          <div aria-hidden className="absolute inset-0 -z-10 overflow-hidden">
            {mode !== "night" ? (
              <>
                <div className="absolute left-[-20%] top-4 h-20 w-60 animate-[cloud_40s_linear_infinite] rounded-[40px] bg-white opacity-70 blur-[0.3px] before:absolute before:left-[-40px] before:top-[10px] before:h-[60px] before:w-[140px] before:rounded-[40px] before:bg-white after:absolute after:right-[-20px] after:top-2 after:h-[60px] after:w-[120px] after:rounded-[40px] after:bg-white" />
                <div className="absolute left-1/3 top-16 h-20 w-60 animate-[cloud_55s_linear_infinite] rounded-[40px] bg-white opacity-60 blur-[0.3px] before:absolute before:left-[-40px] before:top-[10px] before:h-[60px] before:w-[140px] before:rounded-[40px] before:bg-white after:absolute after:right-[-20px] after:top-2 after:h-[60px] after:w-[120px] after:rounded-[40px] after:bg-white" />
                <div className="absolute left-2/3 top-0 h-20 w-60 animate-[cloud_70s_linear_infinite] rounded-[40px] bg-white opacity-50 blur-[0.3px] before:absolute before:left-[-40px] before:top-[10px] before:h-[60px] before:w-[140px] before:rounded-[40px] before:bg-white after:absolute after:right-[-20px] after:top-2 after:h-[60px] after:w-[120px] after:rounded-[40px] after:bg-white" />
              </>
            ) : (
              <div className="absolute inset-0">
                {stars.map((s, i) => (
                  <i
                    key={i}
                    style={{
                      left: `${s.left}%`,
                      top: `${s.top}px`,
                      animationDelay: `${s.delay}s`,
                    }}
                    className="absolute h-0.5 w-0.5 animate-[twinkle_3.5s_ease-in-out_infinite] rounded-full bg-white"
                  />
                ))}
              </div>
            )}
          </div>
        </header>

        {/* Main */}
        <main className="mx-auto w-full max-w-[1100px] gap-4 p-4">
          {mode === "day" && (
            <section className="relative rounded-3xl border-8 p-6 shadow-[0_12px_0_#4a154b,0_28px_40px_rgba(0,0,0,0.35)] bg-[#9e5c33] border-[#3a322b]">
              <div className="mb-3 flex items-center justify-center">
                <div className="px-6 py-3 rounded-lg border-4 border-[#3a322b] bg-[#a8c637] text-[#ffe5c1] text-sm shadow-[inset_0_-6px_0_#6b9930] shadowed">
                  DAY SUMMARY
                </div>
              </div>

              <KPISection
                day={day}
                holdingCost={history[day]?.holding_cost || 0}
                restockCost={history[day]?.restock_cost || 0}
                episodeProfit={episodeProfit}
                history={history}
                footTraffic={footTraffic}
                sales={sales}
                profit={profit}
                currentPrice={currentPrice}
              />

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <div className="grow" />
                <button
                  onClick={goNight}
                  className="text-xs inline-flex items-center gap-2 rounded-2xl px-6 py-3 text-white border-4 border-[#3a322b] bg-[#dec036] shadow-[inset_0_-6px_0_#e28235] active:shadow-[inset_0_2px_0_#e28235] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pink-300 shadowed"
                >
                  NEXT ▶
                </button>
              </div>
            </section>
          )}

          {mode === "night" && (
            <section className="relative rounded-3xl border-8 p-6 shadow-[0_12px_0_#4a154b,0_28px_40px_rgba(0,0,0,0.35)] bg-[#9e5c33] border-[#3a322b] text-pink-100">
              <div className="mb-3 flex items-center justify-center">
                <div className="px-6 py-3 rounded-lg border-4 border-[#3a322b] bg-[#8354b5] text-[#ffe5c1] text-sm shadow-[inset_0_-6px_0_#604794] shadowed">
                  NIGHT PLANNING
                </div>
              </div>

              <div className="grid grid-cols-12 gap-4">
                {/* Restock / Price Controls */}
                <div className="col-span-12 md:col-span-6 rounded-2xl p-4 shadow-[0_8px_0_#683719,0_14px_24px_rgba(0,0,0,0.25)] bg-[#7d4321]">
                  <div className="text-xl shadowed text-[#ffe5c1]">
                    Controls
                  </div>

                  <div className="mt-3 space-y-4">
                    {/* Restock slider */}
                    <div className="flex items-center gap-3">
                      <label className="text-xs w-20" htmlFor="restock-slider">
                        Restock
                      </label>
                      <input
                        id="restock-slider"
                        type="range"
                        min={0}
                        max={maxRestock}
                        value={qty}
                        onChange={(e) => setQty(Number(e.target.value))}
                        className="w-full md:w-64 accent-pink-300"
                        aria-label="Restock quantity slider"
                        aria-describedby="restock-help"
                        disabled={done}
                      />
                      <span className="text-sm">{qty} units</span>
                    </div>

                    {/* Price slider */}
                    <div className="flex items-center gap-3">
                      <label className="text-xs w-20" htmlFor="price-slider">
                        Price
                      </label>
                      <input
                        id="price-slider"
                        type="range"
                        min={Math.max(1, COST_PER_UNIT)}
                        max={MAX_PRICE}
                        value={pricePlan}
                        onChange={(e) => setPricePlan(Number(e.target.value))}
                        className="w-full md:w-64 accent-pink-300"
                        aria-label="Price slider"
                        disabled={done}
                      />
                      <span className="text-sm">${pricePlan.toFixed(2)}</span>
                    </div>

                    <AIRecommendationCard
                      suggestedPrice={aiSuggestedPrice}
                      suggestedQty={aiSuggestedRestockQty}
                      maxRestock={maxRestock}
                      onUse={applyAISuggestion}
                      onUseAndStart={useAndStart}
                      disabled={done}
                    />

                    {/* Current + After restock summary */}
                    <div className="text-xs text-[#ffe5c1]" id="restock-help">
                      <div className="flex items-center gap-2">
                        <span className="shadowed">Cost per unit:</span>
                        <b>${COST_PER_UNIT}</b>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2">
                        <span className="shadowed">Inventory now:</span>
                        <b>{inventoryLeft}</b>
                      </div>
                      <div
                        className="mt-1 flex flex-wrap items-center gap-2"
                        aria-live="polite"
                      >
                        <span className="shadowed">After restock:</span>
                        <b className="">{newInventoryAfterRestock}</b>
                        <span>units</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Inventory batches */}
                <div className="col-span-12 md:col-span-6 rounded-2xl p-4 shadow-[0_8px_0_#683719,0_14px_24px_rgba(0,0,0,0.25)] bg-[#7d4321]">
                  <InventoryBatchVisualizer
                    shelfLife={shelfLife}
                    currentDay={day}
                    batches={batches}
                    inventoryCapacity={inventoryCapacity}
                    inventoryLeft={inventoryLeft}
                  />
                </div>

                {/* Charts */}
                <div className="col-span-12 md:col-span-6 rounded-2xl p-4 shadow-[0_8px_0_#683719,0_14px_24px_rgba(0,0,0,0.25)] bg-[#7d4321]">
                  <div className="bg-[#683719] rounded-xl shadow-md p-4">
                    <h3 className="text-xl font-bold mb-3 shadowed">Sales Over Time</h3>
                    <div style={{ width: "100%", height: "300px" }}>
                      <DemandChart data={history} />
                    </div>
                  </div>
                </div>

                <div className="col-span-12 md:col-span-6 rounded-2xl p-4 shadow-[0_8px_0_#683719,0_14px_24px_rgba(0,0,0,0.25)] bg-[#7d4321]">
                  <div className="bg-[#683719] rounded-xl shadow-md p-4">
                    <h3 className="text-xl font-bold mb-3 shadowed">
                      Revenue Over Time
                    </h3>
                    <div style={{ width: "100%", height: "300px" }}>
                      <RevenueChart data={history} />
                    </div>
                  </div>
                </div>

                <div className="col-span-12 md:col-span-6 rounded-2xl p-4 shadow-[0_8px_0_#683719,0_14px_24px_rgba(0,0,0,0.25)] bg-[#7d4321]">
                  <div className="bg-[#683719] rounded-xl shadow-md p-4">
                    <h3 className="text-xl font-bold mb-3 shadowed">
                      Conversion & Price
                    </h3>
                    <div style={{ width: "100%", height: 300 }}>
                      <ConversionPriceChart data={history} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <div className="grow" />
                <button
                  onClick={goDay}
                  className="text-xs inline-flex items-center gap-2 rounded-2xl px-6 py-3 text-white border-4 border-[#3a322b] bg-[#dec036] shadow-[inset_0_-6px_0_#e28235] active:shadow-[inset_0_2px_0_#e28235] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pink-300 shadowed"
                >
                  ◀ BACK
                </button>
                {!done && (
                  <button
                    onClick={() => nextDay(pricePlan, qty)}
                    className="text-xs inline-flex items-center gap-2 rounded-2xl px-6 py-3 text-white border-4 border-[#3a322b] bg-[#dec036] shadow-[inset_0_-6px_0_#e28235] active:shadow-[inset_0_2px_0_#e28235] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pink-300 shadowed"
                  >
                    START NEXT DAY ▶
                  </button>
                )}
              </div>
            </section>
          )}
        </main>

        <footer className="flex items-center justify-between gap-3 p-4 text-sm" />
      </div>

      <style>{`@keyframes cloud{from{transform:translateX(-20%)}to{transform:translateX(120%)}}@keyframes twinkle{0%,100%{opacity:0}50%{opacity:1}}`}</style>
    </div>
  );
}
