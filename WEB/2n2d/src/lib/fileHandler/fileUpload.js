import { sendCSV, sendModel } from "../2n2dAPI";

function readFileAsArrayBuffer(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(new Error("File reading failed"));
    reader.readAsArrayBuffer(file);
  });
}

function arrayBufferToBase64(buffer) {
  const uint8Array = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < uint8Array.byteLength; i++) {
    binary += String.fromCharCode(uint8Array[i]);
  }
  return btoa(binary);
}

function base64ToBlob(base64, mimeType) {
  const byteChars = atob(base64);
  const byteNumbers = new Array(byteChars.length)
    .fill(0)
    .map((_, i) => byteChars.charCodeAt(i));
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

export async function uploadCSV(ev) {
  const file = ev.target.files[0];
  if (!file) return;

  // const arrayBuffer = await readFileAsArrayBuffer(file);
  // const base64 = arrayBufferToBase64(arrayBuffer);
  //
  // const blob = base64ToBlob(base64, "text/csv");
  const formData = new FormData();
  formData.append("file", file, file.name);

  return await sendCSV(formData);
}

export async function uploadONNX(ev) {
  const file = ev.target.files[0];
  if (!file) return;

  // const arrayBuffer = await readFileAsArrayBuffer(file);
  // const base64 = arrayBufferToBase64(arrayBuffer);
  //
  // const blob = base64ToBlob(base64);
  const formData = new FormData();
  formData.append("file", file, file.name);

  return await sendModel(formData);
}

async function uploadCSVFile(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file, file.name);

  return await sendCSV(formData);
}

async function uploadModelFile(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file, file.name);

  return await sendModel(formData);
}

export async function dragUpload(ev) {
  ev.preventDefault();

  if (ev.dataTransfer.items) {
    //only get the first file, ignore the rest
    for (let i = 0; i < ev.dataTransfer.items.length; i++) {
      const item = ev.dataTransfer.items[i];

      if (item.kind === "file") {
        const file = item.getAsFile();
        console.log(file);

        if (file.name.split(".").pop().toLowerCase() === "csv") {
          return await uploadCSVFile(file);
        } else if (file.name.split(".").pop().toLowerCase() === "onnx") {
          return await uploadModelFile(file);
        }
      }
    }
  }
}
