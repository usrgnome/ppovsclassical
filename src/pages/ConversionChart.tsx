// src/components/ConversionChart.tsx
import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Label,
  Legend,
  ResponsiveContainer,
} from "recharts";

export interface ConversionChartPoint {
  price: number;
  conversionPct: number; // 0..100
}

export default function ConversionChart({ data }: { data: ConversionChartPoint[] }) {
  return (
    <div className="p-6 bg-[#683719] rounded-xl shadow-md h-[100%]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#ffe4c0" strokeDasharray="3 3" />
          <XAxis
            dataKey="price"
            stroke="#ffe4c0"
            tick={{ fill: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
          >
            <Label
              value="Price"
              position="insideBottomRight"
              offset={-5}
              style={{ fill: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
              className="shadowed"
            />
          </XAxis>
          <YAxis
            domain={[0, 100]}
            stroke="#ffe4c0"
            tick={{ fill: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
          >
            <Label
              value="Conversion (%)"
              angle={-90}
              position="insideLeft"
              style={{ fill: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
              className="shadowed"
            />
          </YAxis>
          <Tooltip
            formatter={(v: number) => [`${v.toFixed(1)}%`, "Conversion"]}
            labelFormatter={(label) => `Price: ${label.toFixed(2)}`}
            contentStyle={{
              backgroundColor: "#3a322c",
              border: "1px solid #ffe4c0",
              color: "#ffe4c0",
              fontFamily: "Rowdies, cursive",
            }}
            labelStyle={{ color: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
            itemStyle={{ color: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
          />
          <Legend
            wrapperStyle={{
              color: "#ffe4c0",
              fontFamily: "Rowdies, cursive",
            }}
          />
          <Line
            type="monotone"
            dataKey="conversionPct"
            name="Conversion"
            stroke="#ffe4c0"
            strokeWidth={2}
            dot={{ fill: "#ffe4c0" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
