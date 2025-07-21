"use client";

import Plot from "react-plotly.js";

type CorrelationMatrixProps = {
    matrix: {
        columns: string[];
        index: string[];
        data: number[][];
    };
};

export default function Heatmap({matrix}: CorrelationMatrixProps) {
    const {columns, index, data} = matrix;
    return (
        <div className="w-full overflow-auto flex justify-center items-center">
            <Plot
                data={[
                    {
                        z: data,
                        x: columns,
                        y: index,
                        type: "heatmap",
                        colorscale: [
                            [0, "#232323"],
                            [0.5, "#4f46e5"],
                            [1, "#ffffff"],
                        ],
                        zmin: -1,
                        zmax: 1,
                        showscale: true,
                        hoverongaps: false,
                    },
                ]}
                layout={{
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    margin: {t: 50, b: 80, l: 100, r: 30},
                    xaxis: {
                        tickangle: -45,
                        automargin: true,
                        tickfont: {
                            size: 12,
                            color: "#cccccc",
                        },
                    },
                    yaxis: {
                        tickfont: {
                            size: 12,
                            color: "#cccccc",
                        },
                    },
                }}
                style={{
                    width: "90%",
                    height: "600px",
                    backgroundColor: "transparent",
                    borderRadius: "0.5rem",
                }}
                config={{
                    responsive: true,
                    displayModeBar: false,
                    scrollZoom: false,
                    staticPlot: false,
                }}

            />
        </div>
    );
}
