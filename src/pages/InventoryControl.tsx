import React, { useState, useEffect } from "react";
import DemandChart from "./DemandChart";
import RevenueChart from "./RevenueChart";
import { HistoryEntry } from "../types/types";

import { motion, AnimatePresence } from "framer-motion";

export interface Props {
  done: boolean;
  day: number;
  currentPrice: number;
  inventoryLeft: number;
  shelfLife: number;
  profit: number;
  sales: number;
  history: HistoryEntry[];
  batches: number[][];
  onConfirm: (newPrice: number, restockQty: number) => void;

  // New props for AI recommendation:
  aiSuggestedPrice?: number;
  aiSuggestedRestockQty?: number;
}

function InventoryBatchVisualizer({
  currentDay,
  batches,
  shelfLife,
}: {
  shelfLife: number;
  currentDay: number;
  batches: number[][];
}) {
  // Assuming your existing batch visualizer code here...
  return (
    <div className="bg-white p-4 rounded-xl shadow-md w-full">
      <h3 className="text-xl font-bold mb-3">Inventory Batches</h3>
      {batches.length === 0 ? (
        <p className="text-gray-600">No inventory batches available.</p>
      ) : (
        <div className="overflow-y-auto max-h-[200px] pr-1">
          <ul className="space-y-3">
            {batches.map((batch, index) => {
              const dayCreated = batch[0];
              const expireDay = dayCreated + shelfLife;
              const quantity = batch[1];
              const daysLeft = expireDay - currentDay;
              const percentLeft = Math.max(0, (daysLeft / shelfLife) * 100);

              return (
                <li
                  key={index}
                  className="border border-gray-200 rounded-md p-3 flex flex-col space-y-1"
                >
                  <div className="flex justify-between text-gray-700 font-medium">
                    <span>Batch #{index + 1}</span>
                    <span>{quantity} units</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-3 rounded-full ${
                          percentLeft > 30
                            ? "bg-green-400"
                            : percentLeft > 10
                            ? "bg-yellow-400"
                            : "bg-red-500"
                        }`}
                        style={{ width: `${percentLeft}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500">
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

type SkyCycleProps = {
  day: boolean;
};

function SkyCycle({ day }: SkyCycleProps) {
  const [show, setShow] = useState(day);

  useEffect(() => {
    // Delay the switch for smooth transition
    const timeout = setTimeout(() => setShow(day), 300);
    return () => clearTimeout(timeout);
  }, [day]);

  return (
    <div
      className="w-full h-16 rounded-2xl overflow-hidden shadow-xl relative transition-colors duration-1000"
      style={{
        background: show
          ? "linear-gradient(to top, #87ceeb, #fffde4)"
          : "linear-gradient(to top, #0f2027, #203a43, #2c5364)",
      }}
    >
      <AnimatePresence mode="wait">
        {show ? (
          <motion.div
            key="sun"
            initial={{ opacity: 0, scale: 0.25, x: "-50%", y: "-50%" }}
            animate={{ opacity: 1, scale: 0.5, x: "-50%", y: "-50%" }}
            exit={{ opacity: 0, scale: 0.5, x: "-50%", y: "-50%" }}
            transition={{ duration: 0.5 }}
            className="absolute left-1/2 top-1/2 w-20 h-20 rounded-full bg-yellow-400 shadow-lg"
            style={{
              boxShadow: "0 0 30px rgba(255,255,255,0.7)",
            }}
          />
        ) : (
          <motion.div
            key="moon"
            initial={{ opacity: 0, scale: 0.25, x: "-50%", y: "-50%" }}
            animate={{ opacity: 1, scale: 0.5, x: "-50%", y: "-50%" }}
            exit={{ opacity: 0, scale: 0.5, x: "-50%", y: "-50%" }}
            transition={{ duration: 0.5 }}
            className="absolute left-1/2 top-1/2 w-20 h-20 rounded-full bg-white"
            style={{
              boxShadow: "0 0 30px rgba(255,255,255,0.7)",
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default function InventoryDashboard({
  day,
  done,
  currentPrice,
  inventoryLeft,
  sales,
  history,
  batches,
  profit,
  onConfirm,
  shelfLife,
  aiSuggestedPrice,
  aiSuggestedRestockQty,
}: Props) {
  const [newPrice, setNewPrice] = useState(currentPrice);
  const [restockQty, setRestockQty] = useState(0);

  const handleConfirm = () => {
    if (newPrice <= 0) {
      alert("Price must be greater than zero.");
      return;
    }
    if (restockQty < 0) {
      alert("Restock quantity cannot be negative.");
      return;
    }
    onConfirm(newPrice, restockQty);
    setRestockQty(0);
  };
  const [isDay, setIsDay] = useState(false);

  return (
    <div className="min-h-screen bg-gray-100 px-4 py-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
        {/* LEFT: Controls + Batches */}
        <div className="flex flex-col gap-6 min-h-[620px]">
          {/* Summary + Controls */}
          <section className="bg-white rounded-xl shadow-md p-6 flex flex-col flex-grow">
            <div className="space-y-4 pb-1">
              <SkyCycle day={isDay} />
            </div>

            <header className="mb-4">
              <h2 className="text-3xl font-bold text-gray-900">
                {day === 0 ? "Night Before Sale Period" : `Recap Of Day ${day}`}
              </h2>
              <p className="text-sm text-gray-500 uppercase tracking-wide">
                Inventory Control
              </p>
            </header>


            {day >= 0 && <div className="space-y-4 text-gray-800 text-base flex-grow">
              {[
                { label: "Sales Today", value: `${sales}`, use: day > 0 },
                { label: "Inventory Left", value: `${inventoryLeft} units`, use: true },
                { label: "Profit", value: `$${profit.toFixed(2)}`, use: day > 0},
                {
                  label: "Current Price",
                  value: `$${currentPrice.toFixed(2)}`,
                  use: day > 0
                },
              ].map(({ label, value, use }) => (
                use && <div key={label} className="flex justify-between">
                  <span className="font-medium">{label}</span>
                  <span className="text-indigo-600 font-semibold">{value}</span>
                </div>
              ))}
            </div>}

            {/* AI Recommendation Box */}
            {!done &&
              (aiSuggestedPrice !== undefined ||
                aiSuggestedRestockQty !== undefined) && (
                <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg border border-blue-300 shadow-sm flex items-center space-x-4">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-8 w-8 text-blue-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                  <div>
                    <h3 className="text-lg font-semibold text-blue-700 mb-1">
                      AI Recommendation
                    </h3>
                    <p className="text-blue-800">
                      Suggested Price:{" "}
                      <span className="font-mono">
                        {aiSuggestedPrice !== undefined
                          ? `$${aiSuggestedPrice.toFixed(2)}`
                          : "N/A"}
                      </span>
                    </p>
                    <p className="text-blue-800">
                      Suggested Restock Qty:{" "}
                      <span className="font-mono">
                        {aiSuggestedRestockQty !== undefined
                          ? aiSuggestedRestockQty
                          : "N/A"}
                      </span>
                    </p>
                    <button
                      className="mt-3 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                      onClick={() => {
                        if (
                          aiSuggestedPrice !== undefined &&
                          aiSuggestedRestockQty !== undefined
                        ) {
                          setRestockQty(
                            parseInt(aiSuggestedRestockQty as any, 10)
                          );
                          setNewPrice(parseFloat(aiSuggestedPrice as any));
                        }
                      }}
                    >
                      Use
                    </button>
                  </div>
                </div>
              )}

            {done ? (
              <div className="mt-6 flex flex-col gap-4">
                <button
                  type="submit"
                  className={`mt-2 text-white py-2 rounded-lg font-semibold transition shadow-md
                  ${
                    done
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-indigo-600 hover:bg-indigo-700"
                  }`}
                >
                  {done ? "Finish Simulation" : "Confirm & Next Day"}
                </button>
              </div>
            ) : (
              <form
                className="mt-6 flex flex-col gap-4"
                onSubmit={(e) => {
                  setIsDay(true)
                  e.preventDefault();
                  handleConfirm();
                }}
              >
                <label className="block">
                  <span className="text-sm font-medium text-gray-700">
                    Set Price for Tomorrow ($)
                  </span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    value={newPrice}
                    onChange={(e) => setNewPrice(parseFloat(e.target.value))}
                    className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    required
                  />
                </label>

                <label className="block">
                  <span className="text-sm font-medium text-gray-700">
                    How Much Stock to Order for Tomorrow (units)
                  </span>
                  <input
                    type="number"
                    min="0"
                    value={restockQty}
                    onChange={(e) =>
                      setRestockQty(parseInt(e.target.value, 10))
                    }
                    className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    required
                  />
                </label>

                <button
                  type="submit"
                  className={`mt-2 text-white py-2 rounded-lg font-semibold transition shadow-md bg-indigo-600 hover:bg-indigo-700`}
                >
                  Confirm & Next Day
                </button>
              </form>
            )}
          </section>

          {/* Inventory Batches */}
          <InventoryBatchVisualizer
            shelfLife={shelfLife}
            currentDay={day}
            batches={batches}
          />
        </div>

        {/* RIGHT: Charts */}
        <div className="flex flex-col gap-6">
          <div className="bg-white rounded-xl shadow-md p-4">
            <h3 className="text-xl font-bold text-gray-800 mb-3">
              Demand Over Time
            </h3>
            <div style={{ width: "100%", height: "300px" }}>
              <DemandChart data={history} />
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-md p-4">
            <h3 className="text-xl font-bold text-gray-800 mb-3">
              Revenue Over Time
            </h3>
            <div style={{ width: "100%", height: "300px" }}>
              <RevenueChart data={history} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
