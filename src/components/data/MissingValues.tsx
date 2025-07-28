'use client';

import Plot from 'react-plotly.js';

type missingValuesMatrix = {
  columns: string[];
  data: number[][];
};

export default function MissingDataHeatmap(matrix: missingValuesMatrix) {
  const { columns, data } = matrix;

  return (
    <div className='flex w-full items-center justify-center overflow-auto'>
      <Plot
        data={[
          {
            z: data,
            x: columns,
            y: Array.from({ length: data.length }, (_, i) => i + 1),
            type: 'heatmap',
            colorscale: [
              [0, '#0f973f'],
              [1, '#ef4444']
            ],
            zmin: 0,
            zmax: 1,
            showscale: false,
            hoverongaps: false
          }
        ]}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: { t: 50, b: 80, l: 100, r: 30 },
          xaxis: {
            tickangle: -45,
            automargin: true,
            tickfont: { size: 12, color: '#cccccc' }
          },
          yaxis: {
            tickfont: { size: 10, color: '#999999' },
            title: { text: 'Row', font: { color: '#cccccc' } }
          }
        }}
        style={{
          width: '90%',
          height: '600px',
          backgroundColor: 'transparent',
          borderRadius: '0.5rem'
        }}
        config={{
          responsive: true,
          displayModeBar: false
        }}
      />
    </div>
  );
}
