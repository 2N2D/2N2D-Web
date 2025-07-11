import { getSessionTokenHash } from '@/lib/auth/authentication';
import {
  updateData,
  updateVis,
  updateOptimize,
  updateCsvUrl,
  updateOnnxUrl
} from '@/lib/sessionHandling/sessionUpdater';

const endp = process.env.NEXT_PUBLIC_TWONTWOD_ENDPOINT;

export async function startOptimization(
  selectedInputs: String[],
  targetFeature: String,
  epochs: number,
  sessionId: number,
  csvPath: string,
  onnxPath: string,
  encoding: string
) {
  if (!targetFeature) {
    return 'No target feature selected';
  }

  if (!selectedInputs || selectedInputs.length === 0)
    return 'No input features selected';

  if (!csvPath || !onnxPath) {
    return 'No onnx or csv file selected';
  }

  updateOptimize(sessionId, '', null);

  try {
    const res = await fetch(endp + '/optimize', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'session-id': `${await getSessionTokenHash()}`
      },
      body: JSON.stringify({
        input_features: selectedInputs,
        target_feature: targetFeature,
        max_epochs: epochs,
        session_id: sessionId,
        csv_path: csvPath,
        onnx_path: onnxPath,
        encoding: encoding
      })
    });
    const data = await res.json();
    updateOptimize(sessionId, data.url, data);

    return data;
  } catch (error) {
    console.error('Error during optimization:', error);
  }
}

export async function sendCSV(filePath: string, sessionId: number) {
  try {
    const result = await fetch(endp + '/upload-csv', {
      headers: {
        'session-id': `${await getSessionTokenHash()}`,
        'Content-Type': 'application/json'
      },
      method: 'POST',
      body: JSON.stringify({ filepath: filePath })
    });

    const response = await result.json();
    console.log(response);
    sessionStorage.setItem('csvData', JSON.stringify(response));
    await updateData(sessionId, response);
    await updateCsvUrl(sessionId, filePath);
    return response;
  } catch (error) {
    console.error('Upload failed', error);
  }
}

export async function sendModel(
  filePath: string,
  fileName: string,
  sessionId: number
) {
  try {
    const result = await fetch(endp + '/upload-model', {
      headers: {
        'session-id': `${await getSessionTokenHash()}`,
        'Content-Type': 'application/json'
      },
      method: 'POST',
      body: JSON.stringify({ filepath: filePath })
    });

    const response = await result.json();
    console.log(response);
    sessionStorage.setItem('modelData', JSON.stringify(response));
    sessionStorage.setItem('modelResponse', JSON.stringify(response));
    await updateVis(sessionId, response, fileName);
    await updateOnnxUrl(sessionId, filePath);
    return response;
  } catch (error) {
    console.error('Upload failed', error);
  }
}
