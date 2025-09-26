
const API_URL = "http://127.0.0.1:5002/analyze"; 

export async function uploadFile(file) {
  try {
    const formData = new FormData();
    formData.append("file", file); 

    const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error details:", {
      message: error.message
    });
    throw error;
  }
}

