import React from "react";
import { Card, Typography, Tag, Row, Col } from "antd";
import CoherenceChart from "./charts/CoherenceChart";
import StructureChart from "./charts/StructureChart";
import ReadabilityChart from "./charts/ReadabilityChart";

const { Title, Paragraph } = Typography;

export default function Results({ results }) {
  if (!results) return null;

  const coherenceValue = Math.round(results.coherence_result); // число 0-100
  const coherenceInterpret = results.coherence_interpretation;

  const structureValue = Math.round(results.structure_result);   // число 0-100
  const structureInterpret = results.structure_interpret;

  return (
    <Card style={{ marginTop: 20 }}>
      <Title level={4}>Результаты анализа</Title>

      <Row gutter={16}>
        <Col span={8}>
          <CoherenceChart value={coherenceValue} description={coherenceInterpret} />
        </Col>
        <Col span={8}>
          <StructureChart value={structureValue} description={structureInterpret} />
        </Col>
        <Col span={8}>
          <ReadabilityChart value={results.read_result} />
        </Col>
      </Row>

      <Paragraph style={{ marginTop: 20 }}>
        <b>Ключевые слова:</b>
      </Paragraph>
      {results.keywords?.map((word, idx) => (
        <Tag key={idx} color="blue">
          {word}
        </Tag>
      ))}
    </Card>
  );
}


