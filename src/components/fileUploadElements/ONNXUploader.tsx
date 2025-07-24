import React from 'react';
import { dragUpload, uploadCSV, uploadONNX } from '@/lib/fileHandler/fileUpload';

export default function ONNXUploader({ callBack }: { callBack?: () => void }) {
  const [uploadState, setUploadState] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);

  async function _uploadONNX(e: any) {
    const id = sessionStorage.getItem('currentSessionId');
    if (!id) return;

    setUploading(true);
    const rez = await uploadONNX(e, parseInt(id));
    if (typeof rez === 'string')
      console.log(rez);
    console.log(rez);
    sessionStorage.setItem('modelData', JSON.stringify(rez));
    sessionStorage.setItem('modelResponse', JSON.stringify(rez));

    if (callBack) callBack();
    setUploading(false);
  }

  async function _uploadONNXDrop(e: any) {
    e.preventDefault();
    const _result = await dragUpload(e);
    if (_result == null) return;
    if (callBack) callBack();
  }

  return <>
    <button
      onClick={() => {
        setUploadState(true);
      }}
      className={'uploadButton'}
    >
      {uploading ? 'Uploading...' : <p> Load ONNX File <i className="fa-solid fa-upload"></i></p>}
    </button>
    {uploadState ? (
      <div className={'popup'}>
        <button
          className={'fileDropBack'}
          onClick={() => {
            setUploadState(false);
          }}
        >
          <i className="fa-solid fa-xmark-large"></i>
          Cancel
        </button>
        <div
          className="fileDrop"
          onDrop={(e) => {
            _uploadONNXDrop(e);
            setUploadState(false);
          }}
          onDragOver={(event) => event.preventDefault()}
        >
          <label>
            Upload ONNX Dataset <i className="fa-solid fa-upload"></i>
            <input
              type="file"
              id="ONNX-input"
              accept=".onnx"
              onChange={(e) => {
                _uploadONNX(e);
                setUploadState(false);
              }}
            />
          </label>
          <span>or drag and drop files</span>
        </div>
      </div>
    ) : (
      ''
    )}
  </>;
}