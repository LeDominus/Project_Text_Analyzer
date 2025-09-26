import React from "react";
import { Card, Typography, Tag, Row, Col } from "antd";
import CoherenceChart from "./charts/CoherenceChart";
import StructureChart from "./charts/StructureChart";
import ReadabilityChart from "./charts/ReadabilityChart";

const { Title, Paragraph } = Typography;

export default function Results({ results }) {
  if (!results) return null;

  return (
    <Card style={{ marginTop: 20 }}>
      <Title level={4}>Результаты анализа</Title>

      <Paragraph>
        <b>Стиль текста:</b> {results.style_result}
      </Paragraph>

      <Row gutter={16}>
        <Col span={8}>
          <CoherenceChart value={results.coherence_result} />
        </Col>
        <Col span={8}>
          <StructureChart value={results.structure_result} />
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

