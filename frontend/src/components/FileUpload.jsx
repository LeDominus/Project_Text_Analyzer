import React, { useState } from "react";
import { Upload, Button, message } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import { uploadFile } from "../api/API";

export default function FileUpload({ setResults }) {
  const [loading, setLoading] = useState(false);

  const props = {
    beforeUpload: (file) => {
      const isAllowed =
        file.type === "application/pdf" ||
        file.type ===
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
        file.type === "application/msword";

      if (!isAllowed) {
        message.error("Можно загружать только PDF или Word!");
        return Upload.LIST_IGNORE;
      }
      return true;
    },
    customRequest: async ({ file, onSuccess, onError }) => {
      setLoading(true);
      try {
        const data = await uploadFile(file);
        setResults(data);
        message.success("Файл успешно загружен и проанализирован!");
        onSuccess();
      } catch (err) {
        console.error(err);
        message.error("Ошибка анализа файла");
        onError(err);
      } finally {
        setLoading(false);
      }
    },
  };

  return (
    <Upload {...props} showUploadList={false}>
      <Button
        type="primary"
        icon={<UploadOutlined />}
        loading={loading}
      >
        Загрузить файл
      </Button>
    </Upload>
  );
}

