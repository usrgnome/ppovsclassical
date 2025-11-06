import React from "react";

interface InventoryBatch {
  quantity: number;
  expireDay: number;
  restockDay: number;
  shelfLife: number;
}

interface Props {
  currentDay: number;
  batches: InventoryBatch[];
}

export default function InventoryBatchVisualizer({ currentDay, batches }: Props) {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-md max-w-xl mx-auto">
      <h3 className="text-2xl font-bold mb-4">Inventory Batches</h3>
      {batches.length === 0 ? (
        <p className="text-gray-600">No inventory batches available.</p>
      ) : (
        <ul className="space-y-4">
          {batches.map((batch, index) => {
            const daysLeft = batch.expireDay - currentDay;
            const percentLeft = Math.max(0, (daysLeft / batch.shelfLife) * 100);

            return (
              <li
                key={index}
                className="border border-gray-300 rounded-lg p-4 flex flex-col space-y-2"
              >
                <div className="flex justify-between text-gray-700 font-semibold">
                  <span>Batch #{index + 1}</span>
                  <span>{batch.quantity} units</span>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                    <div
                      className={`h-4 rounded-full ${
                        percentLeft > 30
                          ? "bg-green-400"
                          : percentLeft > 10
                          ? "bg-yellow-400"
                          : "bg-red-500"
                      }`}
                      style={{ width: `${percentLeft}%` }}
                    />
                  </div>
                  <span className="text-sm text-gray-600">
                    {daysLeft > 0
                      ? `${daysLeft} day${daysLeft > 1 ? "s" : ""} left`
                      : "Expired"}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}