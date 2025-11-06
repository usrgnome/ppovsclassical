// ConversionPriceChart.tsx
import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label
} from "recharts";

export default function ConversionPriceChart({ data }: { data: any[] }) {
  return (
    <div className="p-6 bg-[#683719] rounded-xl shadow-md h-[100%]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#ffe4c0" strokeDasharray="3 3" />
          <XAxis dataKey="day" stroke="#ffe4c0" tick={{ fill: "#ffe4c0" }}>
            <Label value="Day" position="insideBottomRight" offset={-5} style={{ fill: "#ffe4c0" }} />
          </XAxis>
          <YAxis yAxisId="left" domain={[0, 1]} stroke="#ffe4c0" tick={{ fill: "#ffe4c0" }}>
            <Label value="Conversion" angle={-90} position="insideLeft" style={{ fill: "#ffe4c0" }} />
          </YAxis>
          <YAxis yAxisId="right" orientation="right" stroke="#ffe4c0" tick={{ fill: "#ffe4c0" }}>
            <Label value="Price" angle={90} position="insideRight" style={{ fill: "#ffe4c0" }} />
          </YAxis>
          <Tooltip contentStyle={{ backgroundColor:"#3a322c", border:"1px solid #ffe4c0", color:"#ffe4c0" }} />
          <Legend wrapperStyle={{ color:"#ffe4c0" }} />
          <Line yAxisId="left" type="monotone" dataKey="conversion" name="Conversion" stroke="#F8F32B" dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="price" name="Price" stroke="#E8F1FF" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
