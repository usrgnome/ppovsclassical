import React, { useEffect, useState } from "react";
import Setup from "./pages/Setup";
import InventoryControlScreen from "./pages/InventoryControl";
import InventoryPricingEnv from "./pages/InventoryPricingEnv";
import { HistoryEntry, SimulationConfig } from "./types/types";
import LoadingScreen from "./pages/Connecting";

import { motion, AnimatePresence } from "framer-motion";

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
  className="w-full h-64 rounded-2xl overflow-hidden shadow-xl relative transition-colors duration-1000"
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
        initial={{ opacity: 0, scale: 0.5, x: '-50%' }}
  animate={{ opacity: 1, scale: 1, x: '-50%' }}
  exit={{ opacity: 0, scale: 0.5, x: '-50%' }}
        transition={{ duration: 0.5 }}
        className="absolute top-4 left-1/2 transform -translate-x-1/2 w-20 h-20 rounded-full bg-yellow-400 shadow-lg"
        style={{
          boxShadow: "0 0 30px rgba(255,255,255,0.7)",
        }}
      />
    ) : (
      <motion.div
        key="moon"
        initial={{ opacity: 0, scale: 0.5, x: '-50%' }}
  animate={{ opacity: 1, scale: 1, x: '-50%' }}
  exit={{ opacity: 0, scale: 0.5, x: '-50%' }}
        transition={{ duration: 0.5 }}
        className="absolute top-4 left-1/2 transform -translate-x-1/2 w-20 h-20 rounded-full bg-white"
        style={{
          boxShadow: "0 0 30px rgba(255,255,255,0.7)",
        }}
      />
    )}
  </AnimatePresence>
</div>
  );
}

export default function App() {
  const [isDay, setIsDay] = useState(true);

  return (
    <div className="p-10 space-y-4">
      <SkyCycle day={isDay} />
      <button
        onClick={() => setIsDay(!isDay)}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow"
      >
        Toggle Day/Night
      </button>
    </div>
  );
}