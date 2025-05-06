"use client";

import React, { useEffect, useRef } from "react";
import {
  Chart,
  ScatterController,
  LinearScale,
  PointElement,
  Tooltip,
  Title,
} from "chart.js";
import { createArchitectureComparisonChart } from "@/lib/feHandler";

// Register required Chart.js components
Chart.register(ScatterController, LinearScale, PointElement, Tooltip, Title);

interface Result {
  layers: number;
  neurons: number;
  test_loss: number;
  r2_score: number;
}

interface Props {
  results: Result[];
}

export default function ArchitectureChart({ results }: Props) {
  const chartRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext("2d");
    if (!ctx) return;

    createArchitectureComparisonChart(results, ctx);
  }, [results]);

  return (
    <div style={{ height: "400px" }}>
      <canvas id="architecture-comparison-chart" ref={chartRef}></canvas>
    </div>
  );
}
