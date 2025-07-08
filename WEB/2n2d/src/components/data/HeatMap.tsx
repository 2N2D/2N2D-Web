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
        <div className="w-full overflow-auto">
            <Plot
                data={[
                    {
                        z: data,
                        x: columns,
                        y: index,
                        type: "heatmap",
                        colorscale: "RdBu",
                        zmin: -1,
                        zmax: 1,
                        showscale: true,
                        hoverongaps: false,
                    },
                ]}
                layout={{
                    title: {text: "Correlation Heatmap"},
                    autosize: true,
                    margin: {t: 50},
                }}
                style={{width: "100%", height: "600px", backgroundColor: "var(--background-color)"}}
                config={{responsive: true}}
            />
        </div>
    );
}
