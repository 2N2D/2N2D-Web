"use client";

import React, { useState } from "react";
import "./styles.css";
import { uploadCSV } from "@/lib/fileHandler/fileUpload";

function Data() {
  const [result, setResult] = useState("");

  const [message, setMessage] = useState("");
  const [rowsNr, setRowsNr] = useState(0);
  const [columnsNr, setColumnsNr] = useState(0);
  const [missingValues, setMissingValues] = useState(0);
  const [fileName, setFileName] = useState("");

  async function _uploadCSV(e: any) {
    const result = await uploadCSV(e);
    setMessage("CSV Uploaded!");
    setRowsNr(result.summary.rows);
    setColumnsNr(result.summary.columns);
    setFileName(result.summary.fileName);
    // let missing = 0;
    // result.summary.missingValues.forEach((value: any) => {
    //   missing += value;
    // });
    // setMissingValues(missing);
  }

  return (
    <div className="page">
      <div className="area">
        <h3 className={"subtitle"}>Dataset Overview</h3>
        <div className="dataSum">
          <div className="info">
            <h1>File</h1>
            <h2>{fileName === "" ? "No file uploaded" : fileName}</h2>
          </div>
          <div className="info">
            <h1>Rows</h1>
            <h2>{result ? "-" : rowsNr}</h2>
          </div>
          <div className="info">
            <h1>Columns</h1>
            <h2>{result ? "-" : columnsNr}</h2>
          </div>
          <div className="info">
            <h1>Missing values</h1>
            <h2>{result ? "-" : missingValues}</h2>
          </div>
        </div>
      </div>

      <div className="area">
        <div className="action-group">
          <input type="file" id="csv-input" accept=".csv" />
          <button id="btn-load-csv" className="data-action-button primary">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M12 15V3M12 3L8 7M12 3L16 7"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M3 15V19C3 20.1046 3.89543 21 5 21H19C20.1046 21 21 20.1046 21 19V15"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            Load CSV File
          </button>
        </div>
        <button
          id="btn-clear-data"
          className="data-action-button danger"
          disabled
        >
          Clear Data
        </button>
      </div>

      <div className="area">
        <div id="data-empty-state" className="data-empty-state">
          <h3>No data loaded</h3>
        </div>

        <div id="data-table-view">
          <table className="data-table" id="csv-table">
            <thead>
              <tr id="csv-headers"></tr>
            </thead>
            <tbody id="csv-body"></tbody>
          </table>

          <div className="pagination-controls">
            <button id="prev-page" disabled>
              &larr; Previous
            </button>
            <span id="page-info">Page 1 of 1</span>
            <button id="next-page" disabled>
              Next &rarr;
            </button>
          </div>
        </div>
      </div>

      <div className="area popup">
        <div className="data-upload-container">
          <div className="file-input-container">
            <label
              htmlFor="csv-input-alt"
              className="file-input-label secondary"
            >
              Upload CSV Dataset
            </label>
            <input
              type="file"
              id="csv-input-alt"
              accept=".csv"
              onChange={(e) => _uploadCSV(e)}
            />
          </div>
        </div>
      </div>
      <p>{JSON.stringify(message)}</p>
    </div>
  );
}

export default Data;
