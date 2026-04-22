import React, { useState } from "react";
import {
  Layout,
  Typography,
  Divider,
  Row,
  Col,
  Card,
  Progress,
  Space,
  List,
  Input,
} from "antd";
import FileUpload from "./components/FileUpload";

const { Header, Content } = Layout;
const { Title, Paragraph } = Typography;
const { TextArea } = Input;

const gradientBackground = "linear-gradient(90deg, #1a3c72, #2a5298)";

function App() {
  const [results, setResults] = useState(null);

  return (
    <Layout style={{ minHeight: "100vh", width: "100vw" }}>
      <Header
        style={{
          height: "120px",
          background: gradientBackground,
          padding: "0 50px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
        }}
      >
        <img
          src="/nsuem_dark.png"
          alt="NSUEM"
          style={{ height: "115px", position: "absolute", left: "20px" }}
        />
        <Title
          level={2}
          style={{
            color: "white",
            margin: 0,
            lineHeight: 1.2,
            textAlign: "center",
          }}
        >
          Анализ учебно-методических материалов
        </Title>
      </Header>

      <Content
        style={{
          padding: "40px 50px",
          width: "100%",
          backgroundColor: "#f0f2f5",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          minHeight: "calc(100vh - 64px - 70px)",
        }}
      >
        {/* Инструкция */}
        <Card
          style={{
            maxWidth: 800,
            width: "100%",
            marginBottom: 30,
            borderRadius: 12,
            boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            background: "white",
            textAlign: "center",
          }}
        >
          <Title
            level={4}
            style={{
              background: gradientBackground,
              WebkitBackgroundClip: "text",
              color: "transparent",
            }}
          >
            Инструкция
          </Title>
          <Paragraph>
            Загрузите PDF документ. Система автоматически анализирует:
          </Paragraph>
          <ul style={{ textAlign: "left", paddingLeft: "40px", color: "#555" }}>
            <li>Читаемость текста</li>
            <li>Стиль документа</li>
            <li>Когерентность и структура</li>
            <li>Ключевые слова и термины</li>
            <li>Рекомендации по улучшению структуры от большой языковой модели</li>
          </ul>
        </Card>

        {/* Загрузка документа */}
        <Card
          style={{
            maxWidth: 600,
            width: "100%",
            marginBottom: 40,
            borderRadius: 12,
            boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            textAlign: "center",
            background: "#ffffff",
          }}
        >
          <Title
            level={5}
            style={{
              background: gradientBackground,
              WebkitBackgroundClip: "text",
              color: "transparent",
            }}
          >
            Загрузите документ
          </Title>
          <Paragraph style={{ color: "#555" }}>PDF файл (.pdf)</Paragraph>
          <FileUpload setResults={setResults} />
        </Card>

        {/* Результаты анализа */}
        {results && (
          <div style={{ width: "100%", maxWidth: 1200 }}>
            <Divider orientation="left">
              <Title level={3} style={{ color: "#1677ff" }}>
                Результаты анализа
              </Title>
            </Divider>

            <Row gutter={[24, 24]} style={{ marginBottom: 30 }}>
              {/* Читаемость */}
              <Col xs={24} md={6}>
                <Card
                  style={{
                    textAlign: "center",
                    borderRadius: 12,
                    boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                  }}
                >
                  <Title level={5}>Читаемость</Title>
                  <Progress
                    type="circle"
                    percent={Math.round(
                      results.read_result["Индекс Флеша (русский)"]
                    )}
                    strokeColor={{
                      "0%": "#ff4d4f",
                      "50%": "#fadb14",
                      "100%": "#52c41a",
                    }}
                    width={100}
                  />
                  <Paragraph style={{ marginTop: 10 }}>
                    {results.read_result["Сложность текста"]}
                  </Paragraph>
                </Card>
              </Col>

              {/* Когерентность */}
              <Col xs={24} md={6}>
                <Card
                  style={{
                    textAlign: "center",
                    borderRadius: 12,
                    boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                  }}
                >
                  <Title level={5}>Когерентность</Title>
                  <Progress
                    type="circle"
                    percent={Math.round(results.coherence_result)}
                    strokeColor={{
                      "0%": "#ff4d4f",
                      "50%": "#ffa940",
                      "100%": "#52c41a",
                    }}
                    width={100}
                  />
                  <Paragraph style={{ marginTop: 10 }}>
                    {results.coherence_interpretation}
                  </Paragraph>
                </Card>
              </Col>

              {/* Структура */}
              <Col xs={24} md={6}>
                <Card
                  style={{
                    textAlign: "center",
                    borderRadius: 12,
                    boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                  }}
                >
                  <Title level={5}>Структура</Title>
                  <Progress
                    type="circle"
                    percent={Math.round(results.structure_result)}
                    strokeColor={{
                      "0%": "#ff4d4f",
                      "50%": "#ffa940",
                      "100%": "#52c41a",
                    }}
                    width={100}
                  />
                  <Paragraph style={{ marginTop: 10 }}>
                    {results.structure_interpret}
                  </Paragraph>
                </Card>
              </Col>
            </Row>

            {/* Детали анализа и ключевые слова */}
            <Row gutter={[24, 24]}>
              <Col xs={24} md={16}>
                <Card
                  style={{
                    borderRadius: 12,
                    boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                    padding: "20px",
                    background: "#ffffff",
                  }}
                >
                  <Title level={3} style={{ color: "#1677ff" }}>
                    Детали анализа читаемости
                  </Title>
                  <List
                    dataSource={Object.entries(results.read_result)}
                    renderItem={([key, value]) => (
                      <List.Item key={key}>
                        <strong>{key}:</strong>{" "}
                        {typeof value === "object"
                          ? JSON.stringify(value)
                          : value.toString()}
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>

              <Col xs={24} md={8}>
                <Card
                  style={{
                    borderRadius: 12,
                    boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                    padding: "20px",
                    background: "#ffffff",
                  }}
                >
                  <Title level={4} style={{ color: "#1677ff" }}>
                    Ключевые слова
                  </Title>
                  <Space wrap>
                    {results.keywords?.length > 0 ? (
                      results.keywords.map((word, idx) => (
                        <span
                          key={idx}
                          style={{
                            background: gradientBackground,
                            color: "white",
                            padding: "6px 12px",
                            borderRadius: "20px",
                            fontWeight: 500,
                            fontSize: "0.9rem",
                            display: "inline-block",
                          }}
                        >
                          {word}
                        </span>
                      ))
                    ) : (
                      <Paragraph>Ключевые слова отсутствуют</Paragraph>
                    )}
                  </Space>
                </Card>
              </Col>
            </Row>

            <Card
              style={{
                borderRadius: 12,
                boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                padding: "20px",
                background: "#ffffff",
                width: "100%",
                marginTop: 20,
              }}
            >
              <Title level={3} style={{ color: "#1677ff" }}>
                Рекомендация по улучшению учебного материала
              </Title>
              <TextArea
                autoSize={{ minRows: 5, maxRows: 100 }}
                readOnly
                style={{ 
                  width: "100%", 
                  resize: "none",
                  overflow: "hidden" 
                }}
                value={results.recommendation || "Не удалось получить рекомендации от LLM"}
              />
            </Card>
          </div>
        )}
      </Content>
    </Layout>
  );
}

export default App;

