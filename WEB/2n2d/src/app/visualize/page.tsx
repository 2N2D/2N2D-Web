"use client";
import React, {useState} from "react";
import "./styles.css";
import {createVisualNetwork2D} from "@/lib/feHandler";
import Styles from "@/components/SideBar.module.css";
import {dragUpload, uploadONNX} from "@/lib/fileHandler/fileUpload";

export default function visualize() {
    const [view3d, setView] = React.useState(false);
    const [result, setResult] = React.useState<any>(null);
    const canvasRef = React.useRef<HTMLDivElement>(null);
    const [uploadState, setUploadState] = useState<boolean>(false);

    async function _uploadONNX(e: any) {
        await uploadONNX(e);
        updateView();
    }

    async function _uploadONNXDrop(e: any) {
        e.preventDefault();
        const _result = await dragUpload(e);
        if (_result == null) return;

        updateView();
    }

    async function updateView() {
        let data = sessionStorage.getItem("modelData");
        if (!data) {
            setResult(null);
            if (!canvasRef.current) return;

            const ctx = canvasRef.current;
            if (ctx) {
                createVisualNetwork2D({nodes: [], edges: []}, ctx);
            }
            return;
        }
        data = JSON.parse(data);
        setResult(data);

        if (!canvasRef.current) return;

        const ctx = canvasRef.current;
        if (ctx) {
            createVisualNetwork2D(data, ctx);
        }
    }

    async function clearData() {
        sessionStorage.removeItem("modelData");
        setResult(null);
        updateView();
    }

    React.useEffect(() => {
        updateView();
        console.log(result);
    }, []);

    return (
        <main className="page">
            <div className="divider">
                <div className="area">
                    <div className="toggleButtons">
                        <button
                            className={view3d ? "toggle-button" : "toggle-button active"}
                            onClick={() => setView(false)}
                        >
                            2D View
                        </button>
                        <button
                            className={view3d ? "toggle-button active" : "toggle-button"}
                            onClick={() => setView(true)}
                        >
                            3D View
                        </button>
                    </div>
                    {view3d ? (
                        <div id="network-3d" className="networkView"></div>
                    ) : (
                        <div id="network-2d" className="networkView " ref={canvasRef}></div>
                    )}
                </div>
                <div className="area">
                    <h3 className={"subtitle"}>Model Details</h3>
                    {result == null ? (
                        "No model loaded"
                    ) : (
                        <div className={"overflow-y-auto max-h-[100vh]"}>
                            <div className={"sArea"}>
                                <h2>Producer: {result.summary.producer}</h2>
                                <h2>IR version: {result.summary.ir_version}</h2>
                            </div>
                            <div className={"sArea"}>
                                <h2>Node count: {result.summary.node_count}</h2>
                            </div>
                        </div>
                    )}
                </div>
                <div className="titleWrapper">
                    <h1 className="title">Network Visualization</h1>
                    <div className="sArea vertical">
                        <div className={"dataArea"}>
                            <button
                                onClick={() => {
                                    setUploadState(true);
                                }}
                                className={"uploadButton"}
                            >
                                Load ONNX File <i className="fa-solid fa-upload"></i>
                            </button>
                            <button className={"deleteButton"} onClick={clearData}>
                                Clear Data <i className="fa-solid fa-trash-xmark"></i>
                            </button>
                        </div>
                    </div>
                </div>
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
                ""
            )}
        </main>
    );
}
