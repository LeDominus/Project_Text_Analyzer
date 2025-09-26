import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function StructureChart({ value }) {
  // value — число в процентах (например, 75.3)
  const data = [
    { name: "Совпадение", percent: value },
    { name: "Различия", percent: 100 - value },
  ];

  return (
    <div style={{ width: "100%", height: 300, textAlign: "center" }}>
      <h4>Сходство структуры текста</h4>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Legend />
          <Bar dataKey="percent" fill="#52c41a" />
        </BarChart>
      </ResponsiveContainer>
      <p><b>{value.toFixed(2)}%</b> совпадения</p>
    </div>
  );
}
