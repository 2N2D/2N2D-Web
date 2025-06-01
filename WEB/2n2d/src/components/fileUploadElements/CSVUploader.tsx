import React from 'react';
import {dragUpload, uploadCSV} from "@/lib/fileHandler/fileUpload";

export default function CSVUploader({callBack}: { callBack?: () => void }) {
    const [uploadState, setUploadState] = React.useState(false);

    async function _uploadCSV(e: any) {
        if (!e.target.files[0]) return;
        await uploadCSV(e);

        if (callBack) callBack();
        e.target.value = "";
    }

    async function _uploadCSVDrop(e: any) {
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
            className={"uploadButton"}
        >
            Load CSV File <i className="fa-solid fa-upload"></i>
        </button>
        {uploadState ? (
            <div className={"popup"}>
                <button
                    className={"fileDropBack"}
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
                        _uploadCSVDrop(e);
                        setUploadState(false);
                    }}
                    onDragOver={(event) => event.preventDefault()}
                >
                    <label>
                        Upload CSV Dataset <i className="fa-solid fa-upload"></i>
                        <input
                            type="file"
                            id="csv-input"
                            accept=".csv"
                            onChange={(e) => {
                                _uploadCSV(e);
                                setUploadState(false);
                            }}
                        />
                    </label>
                    <span>or drag and drop files</span>
                </div>
            </div>
        ) : (
            ""
        )}
    </>;
}