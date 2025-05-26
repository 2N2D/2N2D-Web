"use client";

import React, {useState, useEffect} from "react";
import "./styles.css";
import {dragUpload, uploadCSV} from "@/lib/fileHandler/fileUpload";
import DataTable from "@/components/DataTable";

function Data() {
    const [message, setMessage] = useState("");
    const [rowsNr, setRowsNr] = useState<number | null>(null);
    const [columnsNr, setColumnsNr] = useState<number | null>(null);
    const [fileName, setFileName] = useState<string>("");
    const [missed, setMissed] = useState<number>();
    const [result, setResult] = useState<any>(null);
    const [uploadState, setUploadState] = useState<boolean>(false);

    function handleNewData(_result: any) {
        if (_result == null) return;

        setRowsNr(_result.summary.rows);
        setColumnsNr(_result.summary.columns);
        setFileName(_result.summary.filename);
        setResult(_result);
        let missing = 0;
        for (let key in _result.summary.missingValues) {
            missing += _result.summary.missingValues[key];
        }
        setMissed(missing);
    }

    function clearData() {
        setResult(null);
        sessionStorage.removeItem("csvData");
    }

    async function _uploadCSV(e: any) {
        if (!e.target.files[0]) return;
        const _result = await uploadCSV(e);

        handleNewData(_result);
        e.target.value = "";
    }

    async function _uploadCSVDrop(e: any) {
        e.preventDefault();
        const _result = await dragUpload(e);
        if (_result == null) return;

        handleNewData(_result);
    }

    useEffect(() => {
        const data = sessionStorage.getItem("csvData");
        if (data) {
            handleNewData(JSON.parse(data));
        }
    }, [])

    return (
        <div className="page">
            <div className="area">
                <h3 className={"subtitle"}>Dataset Overview</h3>
                <div className="dataSum">
                    <div className="info">
                        <h1>File</h1>
                        <h2>{result == null ? "No file uploaded" : fileName}</h2>
                    </div>
                    <div className="info">
                        <h1>Rows</h1>
                        <h2>{result == null ? "-" : rowsNr}</h2>
                    </div>
                    <div className="info">
                        <h1>Columns</h1>
                        <h2>{result == null ? "-" : columnsNr}</h2>
                    </div>
                    <div className="info">
                        <h1>Missing values</h1>
                        <h2>{result == null ? "-" : missed}</h2>
                    </div>
                </div>
            </div>

            <div className="area">
                <div className={"dataArea"}>
                    <button
                        onClick={() => {
                            setUploadState(true);
                        }}
                        className={"uploadButton"}
                    >
                        Load CSV File <i className="fa-solid fa-upload"></i>
                    </button>
                    <button className={"deleteButton"} onClick={clearData}>
                        Clear Data <i className="fa-solid fa-trash-xmark"></i>
                    </button>
                </div>
            </div>

            <div className="area tableArea">
                <DataTable result={result}/>
            </div>
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
        </div>
    );
}

export default Data;
