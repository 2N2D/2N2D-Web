import React from 'react';
import { dragUpload, uploadCSV } from '@/lib/fileHandler/fileUpload';
import { Trans } from '@lingui/react/macro';

export default function CSVUploader({ callBack }: { callBack?: () => void }) {
  const [uploadState, setUploadState] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);

  async function _uploadCSV(e: any) {
    const id = sessionStorage.getItem('currentSessionId');
    if (!id) return;
    if (!e.target.files[0]) return;
    setUploading(true);
    const rez = await uploadCSV(e, parseInt(id));
    if (typeof rez === 'string') console.log(rez);
    sessionStorage.setItem('csvData', JSON.stringify(rez));

    if (callBack) callBack();
    e.target.value = '';
    setUploading(false);
  }

  async function _uploadCSVDrop(e: any) {
    e.preventDefault();
    const _result = await dragUpload(e);
    if (_result == null) return;

    if (callBack) callBack();
  }

  return (
    <>
      <button
        onClick={() => {
          setUploadState(true);
        }}
        className={'uploadButton'}
        disabled={uploading}
      >
        {uploading ? (
          'Uploading...'
        ) : (
          <p>
            {' '}
            <Trans>Load CSV File</Trans> <i className='fa-solid fa-upload'></i>
          </p>
        )}
      </button>
      {uploadState ? (
        <div className={'popup'}>
          <button
            className={'fileDropBack'}
            onClick={() => {
              setUploadState(false);
            }}
          >
            <i className='fa-solid fa-xmark-large'></i>
            Cancel
          </button>
          <div
            className='fileDrop'
            onDrop={(e) => {
              _uploadCSVDrop(e);
              setUploadState(false);
            }}
            onDragOver={(event) => event.preventDefault()}
          >
            <label>
              <Trans>Upload CSV Dataset</Trans>{' '}
              <i className='fa-solid fa-upload'></i>
              <input
                type='file'
                id='csv-input'
                accept='.csv'
                onChange={(e) => {
                  _uploadCSV(e);
                  setUploadState(false);
                }}
              />
            </label>
            <span>
              <Trans>or drag and drop files</Trans>
            </span>
          </div>
        </div>
      ) : (
        ''
      )}
    </>
  );
}
