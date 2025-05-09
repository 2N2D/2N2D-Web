"use client";
import React from "react";
import "./styles.css";
import { createVisualNetwork2D } from "@/lib/feHandler";

export default function visualize() {
  const [view3d, setView] = React.useState(false);
  const [result, setResult] = React.useState<any>(null);
  const canvasRef = React.useRef<HTMLDivElement>(null);

  async function updateView() {
    let data = sessionStorage.getItem("modelData");
    if (!data) return;
    data = JSON.parse(data);
    setResult(data);

    if (!canvasRef.current) return;

    const ctx = canvasRef.current;
    if (ctx) {
      createVisualNetwork2D(data, ctx);
    }
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
        </div>
        <div className="titleWrapper">
          <h1 className="title">Network Visualization</h1>
        </div>
      </div>
    </main>
  );
}
