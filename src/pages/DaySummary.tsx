import React from "react";

interface DailySummaryProps {
  day: number;
  sales: number;
  inventoryLeft: number;
  onNextDay: () => void;
}

export default function DailySummaryScreen({
  day,
  sales,
  inventoryLeft,
  onNextDay,
}: DailySummaryProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white shadow-lg rounded-2xl p-8 w-full max-w-md">
        <h2 className="text-2xl font-bold text-center text-green-700 mb-6">
          Day {day} Summary
        </h2>

        <div className="space-y-4 text-gray-800 text-lg">
          <SummaryItem label="Total Sales" value={`$${sales}`} />
          <SummaryItem label="Inventory Remaining" value={`${inventoryLeft} units`} />
        </div>

        <button
          onClick={onNextDay}
          className="mt-6 w-full bg-green-600 text-white py-2 rounded-xl font-semibold hover:bg-green-700 transition"
        >
          Proceed to Next Day
        </button>
      </div>
    </div>
  );
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center border-b pb-2">
      <span className="font-medium">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}
