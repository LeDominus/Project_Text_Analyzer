import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export default function ReadabilityChart({ value }) {
  // value — это число (например, от 0 до 100)
  const data = [
    { name: "Читаемость", score: value },
  ];

  return (
    <div style={{ width: "100%", height: 250, textAlign: "center" }}>
      <h4>Индекс читаемости</h4>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 20, right: 30, left: 50, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis dataKey="name" type="category" />
          <Tooltip />
          <Bar dataKey="score" fill="#1677ff" barSize={30} />
        </BarChart>
      </ResponsiveContainer>
      <p><b>{value}</b> / 100</p>
    </div>
  );
}
