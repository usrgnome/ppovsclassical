export interface HistoryEntry {
  day: number;
  revenue: number;
  demand: number;
  sales: number;
  openingStock: number;
  profit: number;
  episodeProfit: number;
  cost: number;
  holding_cost: number;
  restock_cost: number;
  expired: number;
  restocked: number;
  price: number;
}

export interface InventoryBatch {
  quantity: number;
  expireDay: number;
  restockDay: number;
  shelfLife: number;
}

// New enums/union types for safer config
export type DemandModel =
  | "wtpLogNormal"    // Willingness-to-pay (log-normal) with two-point calibration
  | "logitLogPrice"   // Logistic in log(price) with two-point calibration
  | "elasticity"      // Constant elasticity (ε < 0)
  | "linear"          // Simple linear (legacy)
  | "exponential";    // Simple exp decay (legacy)

export type NoiseModel = "poisson" | "binomial" | "negbin";
export type WeekStart = "Mon" | "Sun";

export interface SimulationConfig {
  // inventory/economics
  holdingCost: number;
  maxPrice: number;
  simulationDays: number;
  restockCost: number;
  initialInventory: number;
  inventoryCapacity: number;
  initialPrice: number;
  shelfLifeDays: number;
  trainAI: boolean;

  demandConfig: {
    /* ---------- Foot traffic (visitors) ---------- */
    traffic: {
      /** Baseline daily visitors before seasonality */
      base: number;
      /** Length-7 multipliers; UI may be Sun..Sat, backend can rotate */
      weekdayMultipliers: number[];
      /** Whether multipliers are given as Mon..Sun or Sun..Sat */
      weekdayStart: WeekStart;
      /** Visitor count noise (std dev, absolute units) */
      visitorNoiseStd: number;
      /** If true, very high prices also suppress traffic (not just conversion) */
      priceTrafficCut: boolean;
      /** Pivot price where suppression begins */
      priceTrafficDelta: number;
    };

    /* ---------- Conversion (price -> buy probability) ---------- */
    conversion: {
      model: DemandModel;

      // Two-point calibration (used by wtpLogNormal & logitLogPrice)
      /** Price point 1 and desired conversion at p1 (0..1) */
      p1?: number;
      c1?: number;
      /** Price point 2 and desired conversion at p2 (0..1) */
      p2?: number;
      c2?: number;

      // Constant elasticity parameters
      /** Reference price, its conversion, and elasticity ε (<0) */
      p0?: number;
      c0?: number;
      elasticity?: number;

      // Legacy/simple models (used only if model === "linear" or "exponential")
      linearPriceCoeff?: number;
      expElasticity?: number;
    };

    /* ---------- Demand noise model ---------- */
    noise: {
      model: NoiseModel;
      /** Overdispersion parameter for NegBin (variance = mean + mean^2/k) */
      negbinK?: number;
    };

    /* ---------- Back-compat (deprecated) ---------- */
    /** @deprecated use traffic.base instead */
    baseDemand?: number;
    /** @deprecated use traffic.weekdayMultipliers instead */
    weekdayMultipliers?: number[];
    /** @deprecated use conversion.model instead */
    demandFunc?: string;
    /** @deprecated use conversion.linearPriceCoeff instead */
    linearPriceCoeff?: number;
    /** @deprecated (not used in new models) */
    normalStdDev?: number;
    /** @deprecated use conversion.expElasticity instead */
    expElasticity?: number;
  };
}
