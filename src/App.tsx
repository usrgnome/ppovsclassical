import React, { useEffect, useMemo, useRef, useState } from "react";
import Setup from "./pages/Setup";
import RetroShopUI from "./pages/game";
import LoadingScreen from "./pages/Connecting";
import InventoryPricingEnv from "./pages/InventoryPricingEnv";
import { HistoryEntry, SimulationConfig } from "./types/types";

// ===== Types for WS messages from the new server =====
type Snapshot = {
  step: number[];
  day: number;              // 0-based
  totalDays: number;
  price: number;
  done: boolean;
  batches: [number, number][]; // [dayCreated, qty]
  expireDays: number;
  inventoryLeft: number;
  inventoryCapacity: number;
  aiPrice: number;
  aiRestock: number;
};

type DayEntry = {
  day: number;
  price: number;
  episodeProfit: number;
  sales: number;
  revenue: number;
  profit: number;
  openingStock: number;
  restocked: number;
  expired: number;
  footTraffic: number;
  // KPIs
  conversion: number; // 0..1
  aov: number;
  sellThrough: number; // 0..1
  waste: number;
};

type InitPayload = {
  snapshot: Snapshot;
  history: DayEntry[];
  footTrafficSeries: number[];
  message?: string;
};
type StepPayload = {
  snapshot: Snapshot;
  history: DayEntry[];
};

type WsEnvelope =
  | { type: "handshake"; payload?: any; meta?: any }
  | { type: "init"; payload: InitPayload; meta?: any }
  | { type: "reset"; payload: InitPayload; meta?: any }
  | { type: "step"; payload: StepPayload; meta?: any }
  | { type: "render"; payload?: { ok: boolean }; meta?: any }
  | { type: "error"; payload: { message: string; code?: string }; meta?: any }
  | { type: string; payload?: any; meta?: any }; // fallback for unknown

function isObject(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object";
}

function safeParse<T = unknown>(raw: any): T | null {
  try {
    return JSON.parse(String(raw)) as T;
  } catch {
    return null;
  }
}

function sumInventory(batches: [number, number][]) {
  return batches.reduce((acc, [, qty]) => acc + (Number(qty) || 0), 0);
}

// ===== App =====
export default function App() {
  const [uiState, setUIState] = useState<"setup" | "summary" | "connecting">(
    "connecting"
  );

  // The env wrapper handles the websocket. We provide onMessage/onOpen/onClose hooks.
  const [env] = useState(
    () =>
      new InventoryPricingEnv(
        () => {},                // onOpen handled below
        () => {},                // onMessage placeholder; we bind later
        () => {},                 // onClose placeholder
        window.location.origin.replace(/^https/, "wss").replace(/^http/, "ws") + (/localhost/.test(window.location.host) ? ":8765" : "/ws")
      )
  );

  // Single source of truth for the game state used by UI
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<HistoryEntry[] | DayEntry[]>([]);
  const [footTrafficSeries, setFootTrafficSeries] = useState<number[]>([]);
  const [aiSuggestedPrice, setAISuggestedPrice] = useState<number | undefined>();
  const [aiSuggestedRestockQty, setAISuggestedRestockQty] = useState<number | undefined>();

  // Derived values
  const day = snapshot?.day ?? 0;
  const totalDays = snapshot?.totalDays ?? 0;
  const done = snapshot?.done ?? false;
  const currentPrice = snapshot?.price ?? 0;
  const batches = snapshot?.batches ?? [];
  const inventoryLeft = useMemo(() => snapshot?.inventoryLeft ?? sumInventory(batches), [snapshot, batches]);

  const salesForDay = useMemo(() => {
    const h = history as DayEntry[];
    return h[day]?.sales ?? 0;
  }, [history, day]);

  const profit = useMemo(() => {
    const h = history as DayEntry[];
    return h[day]?.profit ?? 0;
  }, [history, day]);

  // --- Bind websocket lifecycle once ---
  const mounted = useRef(false);
  useEffect(() => {
    if (mounted.current) return;
    mounted.current = true;

    // onOpen → wait for handshake from server, UI stays "connecting" until then
    env.onOpen = () => {
      // We let the server send "handshake" first; just keep spinner for now
    };

    // onClose → go back to connecting screen (or show an error page if you prefer)
    env.onClose = () => {
      setUIState("connecting");
    };

    // onMessage → robustly handle new schema
    env.onMessage = (msg: MessageEvent) => {
      const parsed = safeParse<WsEnvelope>(msg.data);
      if (!parsed || !isObject(parsed) || typeof parsed.type !== "string") {
        console.warn("Malformed message", msg.data);
        return;
      }

      switch (parsed.type) {
        case "handshake": {
          // Server meta might include state. If state==0 → show setup.
          const serverState = (parsed.meta && (parsed.meta as any).state) ?? 0;
          setUIState(serverState === 0 ? "setup" : "summary");
          break;
        }

        case "init":
        case "reset": {
          const p = parsed.payload as InitPayload;
          console.log(p.snapshot);
          setSnapshot(p.snapshot);
          setHistory(p.history as any);
          setFootTrafficSeries(p.footTrafficSeries ?? []);
          // Clear AI suggestions unless you add a suggest endpoint
          setAISuggestedPrice(p.snapshot.aiPrice);
          setAISuggestedRestockQty(p.snapshot.aiRestock);
          setUIState("summary");
          break;
        }

        case "step": {
          const p = parsed.payload as StepPayload;
          console.log(p.snapshot);
          setSnapshot(p.snapshot);
          setHistory(p.history as any);
          setAISuggestedPrice(p.snapshot.aiPrice);
          setAISuggestedRestockQty(p.snapshot.aiRestock);
          // If your server later returns AI hints in meta, you can read them here:
          // setAISuggestedPrice(parsed.meta?.aiSuggestedPrice)
          // setAISuggestedRestockQty(parsed.meta?.aiSuggestedRestockQty)
          break;
        }

        case "render": {
          // no-op (debug hook)
          break;
        }

        case "error": {
          const err = parsed.payload as { message: string; code?: string };
          console.error("Server error:", err?.code, err?.message);
          // Optionally surface a toast here.
          break;
        }

        default: {
          console.log("Unknown message type:", parsed.type, parsed);
        }
      }
    };
  }, [env]);

  // ---- Setup → init the simulation ----
  const initializeSimulation = (config: SimulationConfig, callback: () => void) => {
    // InventoryPricingEnv should send an "init" payload to the server
    env.initialize(config, callback);
    // UI stays in "connecting" until we get "init"
    setUIState("connecting");
  };

  // ---- Night confirm → step the env ----
  const onConfirm = (newPrice: number, restockQty: number) => {
    if (done) return;
    env.nextDay(newPrice, restockQty);
    // UI updates once "step" arrives
  };

  // ---- Foot traffic array for UI (server calls it footTrafficSeries) ----
  // Some of your UI expects prop name `footTraffic`; map it here:
  const footTraffic = footTrafficSeries;

  return (
    <div className="min-h-screen font-rowdies bg-gray-50">
      <div className="mx-auto">
        {uiState === "setup" && (
          <Setup onStart={initializeSimulation} callback={() => {}} />
        )}

        {uiState === "connecting" && <LoadingScreen message="Connecting" />}

        {uiState === "summary" && snapshot && (
          <RetroShopUI
            // from snapshot
            day={day}
            done={done}
            inventoryCapacity={snapshot.inventoryCapacity}
            maxRestock={Math.max(-10, snapshot.inventoryCapacity - snapshot.inventoryLeft)}
            episodeProfit={(history as DayEntry[])[day]?.episodeProfit ?? 0}
            totalDays={totalDays}
            currentPrice={currentPrice}
            batches={batches as unknown as number[][]}
            shelfLife={snapshot.expireDays}
            inventoryLeft={inventoryLeft}
            // from history
            sales={(history as DayEntry[])[day]?.sales ?? 0}
            profit={(history as DayEntry[])[day]?.profit ?? 0}
            history={history as unknown as HistoryEntry[]}
            // series
            footTraffic={footTraffic}
            // actions
            onConfirm={onConfirm}
            // optional suggestions (if you later add them to server meta)
            aiSuggestedPrice={aiSuggestedPrice}
            aiSuggestedRestockQty={aiSuggestedRestockQty}
          />
        )}
      </div>
    </div>
  );
}
