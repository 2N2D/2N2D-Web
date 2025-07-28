'use server';
import {
  getCurrentUser,
  getSessionTokenHash,
  hash
} from '@/lib/auth/authentication';
import {
  updateData,
  updateVis,
  updateOptimize,
  updateCsvUrl,
  updateOnnxUrl
} from '@/lib/sessionHandling/sessionUpdater';

async function sendRequest(body: any, endpoint: string) {
  const request = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'session-id': `${await getCurrentUser()}`,
      'x-api-key': await hash(process.env.TWONTWOD_API_KEY!)
    },
    body: JSON.stringify(body)
  };
  try {
    const res = await fetch(process.env.TWONTWOD_ENDPOINT + endpoint, request);
    return await res.json();
  } catch (error) {
    console.error(error);
  }
}

export async function startOptimization(
  selectedInputs: String[],
  targetFeature: String,
  epochs: number,
  sessionId: number,
  csvPath: string,
  onnxPath: string,
  encoding: string,
  strat: string
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
    const data = await sendRequest(
      {
        input_features: selectedInputs,
        target_feature: targetFeature,
        max_epochs: epochs,
        session_id: sessionId,
        csv_path: csvPath,
        onnx_path: onnxPath,
        encoding: encoding,
        strategy: strat
      },
      '/optimize'
    );

    updateOptimize(sessionId, data.url, data);

    return data;
  } catch (error) {
    console.error('Error during optimization:', error);
  }
}

export async function sendCSV(filePath: string, sessionId: number) {
  try {
    const response = await sendRequest({ filepath: filePath }, '/upload-csv');

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
    const response = await sendRequest({ filepath: filePath }, '/upload-model');
    await updateVis(sessionId, response, fileName);
    await updateOnnxUrl(sessionId, filePath);
    return response;
  } catch (error) {
    console.error('Upload failed', error);
  }
}

export async function getOptimizationStatus(sessionId: string) {
  return `${process.env.TWONTWOD_ENDPOINT}/optimization-status/${sessionId}`;
}
