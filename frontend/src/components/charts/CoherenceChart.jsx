import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

export default function CoherenceChart({ value }) {
  const data = [
    { name: "Когерентность", value: value },
    { name: "Шум", value: 100 - value },
  ];

  const COLORS = ["#4caf50", "#f44336"];

  return (
    <div style={{ textAlign: "center", marginTop: 20 }}>
      <h4>Когерентность текста</h4>
      <PieChart width={300} height={200}>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          outerRadius={70}
          dataKey="value"
          label
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </div>
  );
}
