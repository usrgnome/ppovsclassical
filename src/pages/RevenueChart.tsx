import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Label,
} from "recharts";

interface RevenueChartProps {
  data: { day: number; revenue: number }[];
}

export default function RevenueChart({ data }: RevenueChartProps) {
  return (
    <div className="p-6 bg-[#683719] rounded-xl shadow-md h-[100%]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#ffe4c0" strokeDasharray="3 3" />
          <XAxis
            dataKey="day"
            stroke="#F8F32B"
            tick={{ fill: "#F8F32B", fontFamily: "Rowdies, cursive" }}
          >
            <Label
              value="Day"
              position="insideBottomRight"
              offset={-5}
              style={{ fill: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
              className="shadowed"
            />
          </XAxis>
          <YAxis
            stroke="#F8F32B"
            tick={{ fill: "#F8F32B", fontFamily: "Rowdies, cursive" }}
          >
            <Label
              value="Revenue ($)"
              angle={-90}
              position="insideLeft"
              style={{ fill: "#ffe4c0", fontFamily: "Rowdies, cursive" }}
              className="shadowed"
            />
          </YAxis>
          <Tooltip
            formatter={(value: number) => `$${value.toFixed(2)}`}
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
            dataKey="revenue"
            stroke="#E8F1FF"
            strokeWidth={2}
            dot={{ fill: "#E8F1FF" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
