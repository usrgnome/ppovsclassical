import React, { useMemo, useState } from "react";
import { SimulationConfig } from "../types/types";
import ConversionChart, { ConversionChartPoint } from "./ConversionChart";

type DemandModel =
  | "wtpLogNormal"
  | "logitLogPrice"
  | "elasticity"
  | "linear"
  | "exponential";
type NoiseModel = "poisson" | "binomial" | "negbin";
type WeekStart = "Mon" | "Sun";

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function normalInvCDF(p: number): number {
  // Rational approximation (Acklam) — good enough for UI sampling
  // https://web.archive.org/web/20150910044758/http://home.online.no/~pjacklam/notes/invnorm/
  if (p <= 0 || p >= 1) return p === 0 ? -Infinity : Infinity;
  const a1 = -39.69683028665376,
    a2 = 220.9460984245205,
    a3 = -275.9285104469687;
  const a4 = 138.357751867269,
    a5 = -30.66479806614716,
    a6 = 2.506628277459239;
  const b1 = -54.47609879822406,
    b2 = 161.5858368580409,
    b3 = -155.6989798598866;
  const b4 = 66.80131188771972,
    b5 = -13.28068155288572;
  const c1 = -0.007784894002430293,
    c2 = -0.3223964580411365,
    c3 = -2.400758277161838;
  const c4 = -2.549732539343734,
    c5 = 4.374664141464968,
    c6 = 2.938163982698783;
  const d1 = 0.007784695709041462,
    d2 = 0.3224671290700398,
    d3 = 2.445134137142996,
    d4 = 3.754408661907416;
  const pl = 0.02425,
    ph = 1 - pl;
  let q, r;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    );
  }
  if (ph < p) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    );
  }
  q = p - 0.5;
  r = q * q;
  return (
    ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
    (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
  );
}

function convAtPrice(
  model:
    | "wtpLogNormal"
    | "logitLogPrice"
    | "elasticity"
    | "linear"
    | "exponential",
  price: number,
  params: {
    p1?: number;
    c1?: number;
    p2?: number;
    c2?: number;
    p0?: number;
    c0?: number;
    elasticity?: number;
    linearPriceCoeff?: number;
    expElasticity?: number;
    priceRange?: [number, number];
  }
): number {
  const p = Math.max(1e-9, price);
  if (model === "wtpLogNormal") {
    const p1 = params.p1 ?? 5,
      c1 = clamp01(params.c1 ?? 0.4);
    const p2 = params.p2 ?? 15,
      c2 = clamp01(params.c2 ?? 0.1);
    const z1 = normalInvCDF(1 - c1);
    const z2 = normalInvCDF(1 - c2);
    const denom = z2 - z1 || 1e-9;
    const sigma = (Math.log(p2) - Math.log(p1)) / denom;
    const mu = Math.log(p1) - sigma * z1;
    const x = (Math.log(p) - mu) / sigma;
    // 1 - Phi(x) using erf
    const phi = 0.5 * (1 + erf(x / Math.SQRT2));
    return clamp01(1 - phi);
  }
  if (model === "logitLogPrice") {
    const p1 = params.p1 ?? 5,
      c1 = clamp01(params.c1 ?? 0.4);
    const p2 = params.p2 ?? 15,
      c2 = clamp01(params.c2 ?? 0.1);
    const x1 = Math.log(p1),
      x2 = Math.log(p2);
    const L1 = Math.log(c1 / (1 - c1 + 1e-12));
    const L2 = Math.log(c2 / (1 - c2 + 1e-12));
    const beta = -(L2 - L1) / (x2 - x1 || 1e-12);
    const alpha = -L1 - beta * x1;
    const x = Math.log(p);
    return clamp01(1 / (1 + Math.exp(alpha + beta * x)));
  }
  if (model === "elasticity") {
    const p0 = params.p0 ?? 10,
      c0 = clamp01(params.c0 ?? 0.2);
    const eps = params.elasticity ?? -1.5;
    return clamp01(c0 * Math.pow(p / Math.max(1e-9, p0), eps));
  }
  if (model === "linear") {
    const [lo, hi] = params.priceRange ?? [1, 20];
    const k = params.linearPriceCoeff ?? 1.0;
    return clamp01(1 - k * ((p - lo) / Math.max(1e-9, hi - lo)));
  }
  if (model === "exponential") {
    const a = params.expElasticity ?? 0.05;
    return clamp01(Math.exp(-a * p));
  }
  return 0;
}

// error function helper for WTP curve
function erf(x: number): number {
  // Abramowitz–Stegun approximation
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592,
    a2 = -0.284496736,
    a3 = 1.421413741,
    a4 = -1.453152027,
    a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y =
    1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

export default function IntroScreen({
  onStart,
  callback,
}: {
  onStart: (config: SimulationConfig, callback: () => void) => void;
  callback: () => void;
}) {
  // --- Inventory & costs ---
  const [restockCost, setRestockCost] = useState(2.2);
  const [initialInventory, setInitialInventory] = useState(30);
  const [inventoryCapacity, setInventoryCapacity] = useState(220);
  const [initialPrice, setInitialPrice] = useState(3.5);
  const [maxPrice, setMaxPrice] = useState(6.0);
  const [simulationDays, setSimulationDays] = useState(30);
  const [holdingCost, setHoldingCost] = useState(0.01);
  const [shelfLifeDays, setShelfLifeDays] = useState(7);
  const [trainAI, setTrainAI] = useState(false);

  // --- Foot traffic ---
  const [weekdayMultipliers, setWeekdayMultipliers] = useState<number[]>([
    0.9,0.9,1.0,1.0,1.1,1.3,1.4
  ]); // Sun..Sat by default in UI
  const [weekdayStart, setWeekdayStart] = useState<WeekStart>("Mon"); // align with backend
  const [baseVisitors, setBaseVisitors] = useState(50);
  const [visitorNoiseStd, setVisitorNoiseStd] = useState(40); // default ~ 12% of base 50
  const [priceTrafficCut, setPriceTrafficCut] = useState(false);
  const [priceTrafficDelta, setPriceTrafficDelta] = useState(15); // pivot price where traffic begins to fall

  // --- Conversion ---
  const [convModel, setConvModel] = useState<DemandModel>("wtpLogNormal");
  // two-point calibration (shared)
  const [p1, setP1] = useState(3.0);
  const [c1, setC1] = useState(0.1);
  const [p2, setP2] = useState(5);
  const [c2, setC2] = useState(0.04);
  // elasticity model
  const [p0, setP0] = useState(10);
  const [c0, setC0] = useState(0.2);
  const [elasticity, setElasticity] = useState(-0.5);
  // legacy extras if you keep them
  const [linearPriceCoeff, setLinearPriceCoeff] = useState(1.0);
  const [expElasticity, setExpElasticity] = useState(0.05);

  // --- Demand noise ---
  const [noiseModel, setNoiseModel] = useState<NoiseModel>("poisson");
  const [negbinK, setNegbinK] = useState(5);

  const updateWeekdayMultiplier = (index: number, value: number) => {
    const next = [...weekdayMultipliers];
    next[index] = value;
    setWeekdayMultipliers(next);
  };

  const handleStart = () => {
    const demandConfig = {
      traffic: {
        base: baseVisitors,
        weekdayMultipliers,
        weekdayStart,
        visitorNoiseStd,
        priceTrafficCut, // enable price→traffic coupling
        priceTrafficDelta, // pivot price for coupling
      },
      conversion: {
        model: convModel,
        // two-point parameters used by wtp/logit; ignored otherwise
        p1,
        c1,
        p2,
        c2,
        // elasticity reference
        p0,
        c0,
        elasticity,
        // legacy (used only if model === "linear"/"exponential")
        linearPriceCoeff,
        expElasticity,
      },
      noise: {
        model: noiseModel,
        negbinK,
      },
    };

    onStart(
      {
        restockCost,
        initialInventory,
        inventoryCapacity,
        simulationDays,
        initialPrice,
        maxPrice,
        shelfLifeDays,
        trainAI,
        holdingCost,
        demandConfig,
      } as SimulationConfig,
      callback
    );
  };

  // compute conversion curve data between minPrice and maxPrice
  const minPrice = 1; // or make this a state if you want to expose it
  const convData: ConversionChartPoint[] = useMemo(() => {
    const steps = 60;
    const arr: ConversionChartPoint[] = [];
    for (let i = 0; i <= steps; i++) {
      const price = minPrice + (i * (maxPrice - minPrice)) / steps;
      const conv = convAtPrice(convModel, price, {
        p1,
        c1,
        p2,
        c2,
        p0,
        c0,
        elasticity,
        linearPriceCoeff,
        expElasticity,
        priceRange: [minPrice, maxPrice],
      });
      arr.push({ price, conversionPct: conv * 100 });
    }
    return arr;
    // include all deps that affect conversion
  }, [
    convModel,
    p1,
    c1,
    p2,
    c2,
    p0,
    c0,
    elasticity,
    linearPriceCoeff,
    expElasticity,
    maxPrice,
  ]);

  return (
    <div className="min-h-screen font-rowdies text-[#ffe5c1] relative bg-gradient-to-b from-[#7e2de0] via-[#ff63c3] to-[#ffb86b]">
      <div className="pointer-events-none fixed inset-0 mix-blend-multiply opacity-40 [background:repeating-linear-gradient(0deg,rgba(255,255,255,0.08)_0px,rgba(255,255,255,0.08)_1px,transparent_2px,transparent_3px)]" />
      <div className="mx-auto max-w-[80%] px-4 py-10">
        <div className="rounded-3xl border-8 border-[#3a322b] bg-[#7d4321] shadow-[0_12px_0_#3a322b,0_28px_40px_rgba(0,0,0,0.35)] p-6">
          <div className="mb-6 flex items-center justify-center">
            <div className="px-6 py-3 rounded-lg border-4 border-[#3a322b] bg-[#cf7830] shadow-[inset_0_-6px_0_#a55612] shadowed">
              SIMULATION SETUP v1.0
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Inventory & Costs */}
            <Panel title="Inventory & Costs">
              <InputField
                label="Restock Cost (per unit)"
                value={restockCost}
                onChange={setRestockCost}
                step={0.1}
                min={1}
              />
              <InputField
                label="Initial Stock"
                value={initialInventory}
                onChange={setInitialInventory}
              />
              <InputField
                label="Inventory Capacity"
                value={inventoryCapacity}
                onChange={setInventoryCapacity}
                min={1}
              />
              <InputField
                label="Holding Cost (per unit/day)"
                value={holdingCost}
                onChange={setHoldingCost}
                step={0.1}
              />
              <InputField
                label="Initial Price"
                value={initialPrice}
                onChange={setInitialPrice}
                step={0.1}
                min={1}
              />
              <InputField
                label="Max Price"
                step={0.1}
                value={maxPrice}
                onChange={setMaxPrice}
                min={2}
              />
            </Panel>

            {/* Timing */}
            <Panel title="Pricing & Shelf Life">
              <InputField
                label="Product Shelf Life (days)"
                value={shelfLifeDays}
                onChange={setShelfLifeDays}
                min={1}
              />
              <InputField
                label="Simulation Duration (days)"
                value={simulationDays}
                onChange={setSimulationDays}
                min={1}
              />
            </Panel>

            {/* Foot Traffic */}
            <Panel title="Foot Traffic">
              <InputField
                label="Base Visitors"
                value={baseVisitors}
                onChange={setBaseVisitors}
                min={1}
              />
              <InputField
                label="Visitor Noise Std. Dev."
                value={visitorNoiseStd}
                onChange={setVisitorNoiseStd}
                min={1}
              />
              <div className="grid grid-cols-7 gap-2">
                {weekdayMultipliers.map((val, idx) => (
                  <div key={idx} className="flex flex-col items-center">
                    <span className="text-xs opacity-90 shadowed">
                      {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][idx]}
                    </span>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      className="mt-1 w-full rounded-2xl border-4 border-[#3a322b] bg-[#683719] px-2 py-1 text-center shadow-[inset_0_-6px_0_#4a154b] focus:outline-none text-[#ffe5c1]"
                      value={val}
                      onChange={(e) =>
                        updateWeekdayMultiplier(idx, toNum(e.target.value))
                      }
                    />
                  </div>
                ))}
              </div>
              <div className="mt-2 flex items-center gap-3">
                <input
                  type="checkbox"
                  id="ptc"
                  checked={priceTrafficCut}
                  onChange={(e) => setPriceTrafficCut(e.target.checked)}
                  className="h-5 w-5 accent-[#cf7830]"
                />
                <label htmlFor="ptc" className="shadowed">
                  High prices reduce foot traffic
                </label>
              </div>
              {priceTrafficCut && (
                <InputField
                  label="Traffic Pivot Price (start of suppression)"
                  value={priceTrafficDelta}
                  onChange={setPriceTrafficDelta}
                />
              )}
            </Panel>

            {/* Conversion */}
            <Panel title="Conversion Model">
              <LabeledGroup label="Model">
                <select
                  className="w-full rounded-2xl border-4 border-[#3a322b] bg-[#683719] px-3 py-2 shadow-[inset_0_-6px_0_#4a154b] focus:outline-none text-[#ffe5c1]"
                  value={convModel}
                  onChange={(e) => setConvModel(e.target.value as DemandModel)}
                >
                  <option value="wtpLogNormal">WTP (Log-Normal)</option>
                  <option value="logitLogPrice">Logit on log-price</option>
                  <option value="elasticity">Constant Elasticity</option>
                  <option value="linear">Linear (simple)</option>
                  <option value="exponential">Exponential Decay</option>
                </select>
              </LabeledGroup>

              {/* shared two-point params */}
              {(convModel === "wtpLogNormal" ||
                convModel === "logitLogPrice") && (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <InputField
                      label="p1 (price)"
                      value={p1}
                      onChange={setP1}
                      step={0.1}
                    />
                    <InputField
                      label="c1 (conversion @ p1)"
                      value={c1}
                      onChange={setC1}
                      step={0.01}
                    />
                    <InputField
                      label="p2 (price)"
                      value={p2}
                      onChange={setP2}
                      step={0.1}
                    />
                    <InputField
                      label="c2 (conversion @ p2)"
                      value={c2}
                      onChange={setC2}
                      step={0.01}
                    />
                  </div>
                </>
              )}

              {convModel === "elasticity" && (
                <div className="grid grid-cols-3 gap-3">
                  <InputField
                    label="p0 (reference price)"
                    value={p0}
                    onChange={setP0}
                  />
                  <InputField
                    label="c0 (conv @ p0)"
                    value={c0}
                    onChange={setC0}
                  />
                  <InputField
                    label="Elasticity ε (<0)"
                    value={elasticity}
                    onChange={setElasticity}
                  />
                </div>
              )}

              {convModel === "linear" && (
                <InputField
                  label="Linear Price Coefficient"
                  value={linearPriceCoeff}
                  onChange={setLinearPriceCoeff}
                />
              )}
              {convModel === "exponential" && (
                <InputField
                  label="Exponential Elasticity (α)"
                  value={expElasticity}
                  onChange={setExpElasticity}
                />
              )}

              <div className="h-64 mt-2">
                <ConversionChart data={convData} />
              </div>
            </Panel>

            {/* Demand Noise + AI toggle */}
            <Panel title="Demand Noise & Training">
              <LabeledGroup label="Noise Model">
                <select
                  className="w-full rounded-2xl border-4 border-[#3a322b] bg-[#683719] px-3 py-2 shadow-[inset_0_-6px_0_#4a154b] focus:outline-none text-[#ffe5c1]"
                  value={noiseModel}
                  onChange={(e) => setNoiseModel(e.target.value as NoiseModel)}
                >
                  <option value="poisson">Poisson</option>
                  <option value="binomial">Binomial</option>
                  <option value="negbin">Negative Binomial</option>
                </select>
              </LabeledGroup>
              {noiseModel === "negbin" && (
                <InputField
                  label="NegBin k (overdispersion)"
                  value={negbinK}
                  onChange={setNegbinK}
                />
              )}
              <div className="mt-4 flex items-center gap-3">
                <input
                  type="checkbox"
                  id="trainAI"
                  checked={trainAI}
                  onChange={(e) => setTrainAI(e.target.checked)}
                  className="h-5 w-5 accent-[#cf7830]"
                />
                <label htmlFor="trainAI" className="shadowed">
                  Train AI Model
                </label>
              </div>
            </Panel>
          </div>

          <div className="mt-6">
            <button
              onClick={handleStart}
              className="
                w-full text-xs inline-flex items-center justify-center gap-2
                rounded-2xl px-6 py-3 text-white
                border-4 border-[#3a322b]
                bg-[#dec036]
                shadow-[inset_0_-8px_0_#e28235]
                active:shadow-[inset_0_2px_0_#e28235]
                focus:outline-none
                shadowed
              "
            >
              START SIMULATION ▶
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- UI helpers ---------- */

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl p-4 bg-[#7d4321] border-4 border-[#3a322b] shadow-[0_8px_0_#3a322b,0_14px_24px_rgba(0,0,0,0.25)]">
      <div className="mb-3 flex items-center justify-center">
        <div className="px-4 py-2 rounded-lg border-4 border-[#3a322b] bg-[#a8c637] shadow-[inset_0_-6px_0_#6b9930] shadowed text-sm">
          {title}
        </div>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function LabeledGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1 shadowed">{label}</div>
      {children}
    </div>
  );
}

function InputField({
  label,
  value,
  onChange,
  min = 0,
  step = 'any',
}: {
  label: string;
  value: number;
  onChange: (val: number) => void;
  min?: number;
  step?: number | 'any';
}) {
  const handleBlur = () => { // @ts-ignore
    if (value === '' || isNaN(value)) { // @ts-ignore
      onChange(0); // Default to 0 if input is empty or invalid
    }
  };

  return (
    <div>
      <label className="mb-1 block shadowed">{label}</label>
      <input
        type="number"
        value={Number.isFinite(value) ? value : ''}
        min={min}
        step={step}
        onChange={(e) => {
          const newValue = e.target.value === '' ? '' : parseFloat(e.target.value); // @ts-ignore
          if (!isNaN(newValue)) { // @ts-ignore
            onChange(newValue);
          }
        }}
        onBlur={handleBlur} // Trigger onBlur event
        className="
          w-full rounded-2xl border-4 border-[#3a322b]
          bg-[#683719] px-3 py-2 text-[#ffe5c1]
          shadow-[inset_0_-6px_0_#4a154b]
          focus:outline-none
        "
      />
    </div>
  );
}

function toNum(v: string): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}
