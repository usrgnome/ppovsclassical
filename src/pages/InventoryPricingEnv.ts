// pages/InventoryPricingEnv.ts
import { SimulationConfig } from "../types/types";

/**
 * Thin WebSocket client for the inventory sim server.
 * Contract:
 *  - Server emits envelopes: { type: "handshake" | "init" | "reset" | "step" | "render" | "error", payload, meta }
 *  - Client sends:
 *      { type: "init", payload: SimulationConfig }
 *      { type: "step", payload: { price: number, restock: number } }
 *      { type: "reset" }
 *      { type: "render" }
 */
export default class InventoryPricingEnv {
  // Public (you can reassign these from App.tsx after construction)
  public onOpen: () => void;
  public onMessage: (msg: MessageEvent) => void;
  public onClose: () => void;
  public onError: (ev: Event) => void;

  // Connection
  private ws: WebSocket | null = null;
  private readonly url: string;
  private readonly queue: string[] = [];

  constructor(
    onopen: () => void,
    onmessage: (msg: MessageEvent) => void,
    onclose: () => void,
    url: string = "ws://localhost:8765"
  ) {
    this.url = url;

    // default callbacks (can be replaced by caller later)
    this.onOpen = onopen ?? (() => {});
    this.onMessage = onmessage ?? (() => {});
    this.onClose = onclose ?? (() => {});
    this.onError = () => {};

    this.connect();
  }

  // ---- Lifecycle ------------------------------------------------------------

  private connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.flushQueue();
      this.onOpen();
    };

    // Forward raw MessageEvent to the app (App.tsx does parsing)
    this.ws.onmessage = (msg) => {
      this.onMessage(msg);
    };

    this.ws.onclose = () => {
      this.onClose();
    };

    this.ws.onerror = (ev) => {
      this.onError(ev);
    };
  }

  public close() {
    try {
      this.ws?.close();
    } catch {
      // ignore
    } finally {
      this.ws = null;
    }
  }

  // ---- Sending helpers ------------------------------------------------------

  private send(obj: unknown) {
    const payload = JSON.stringify(obj);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
    } else {
      // queue until open
      this.queue.push(payload);
    }
  }

  private flushQueue() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    while (this.queue.length) {
      const msg = this.queue.shift();
      if (msg) this.ws.send(msg);
    }
  }

  // ---- Public API used by App.tsx -------------------------------------------

  /**
   * Initialize the simulation on the server.
   * @param config Simulation configuration collected from Setup screen
   */
  public initialize(config: SimulationConfig, callback?: () => void) {
    // Server expects: { type: "init", payload: config }
    this.send({ type: "init", payload: config });
    if (callback) callback();
  }

  /**
   * Advance one day with the chosen controls.
   */
  public nextDay(price: number, restock: number) {
    // Server expects: { type: "step", payload: { price, restock } }
    this.send({ type: "step", payload: { price, restock } });
  }

  /**
   * Reset the environment (server will reply with a fresh init-like payload).
   */
  public reset() {
    this.send({ type: "reset" });
  }

  /**
   * Ask server to render (debug).
   */
  public render() {
    this.send({ type: "render" });
  }
}
