import { Chart } from "chart.js";

export async function createArchitectureComparisonChart(results, ctx) {
  const dataByLayers = {};

  results.forEach((result) => {
    const layers = result.layers;
    if (!dataByLayers[layers]) {
      dataByLayers[layers] = [];
    }
    dataByLayers[layers].push({
      neurons: result.neurons,
      loss: result.test_loss,
      r2: result.r2_score,
    });
  });

  const datasets = [];
  const colors = [
    "rgba(54, 162, 235, 0.7)",
    "rgba(255, 99, 132, 0.7)",
    "rgba(75, 192, 192, 0.7)",
  ];

  Object.keys(dataByLayers).forEach((layers, index) => {
    datasets.push({
      label: `${layers} Layer${layers > 1 ? "s" : ""}`,
      data: dataByLayers[layers].map((item) => ({
        x: item.neurons,
        y: item.loss,
      })),
      backgroundColor: colors[index % colors.length],
      borderColor: colors[index % colors.length].replace("0.7", "1"),
      borderWidth: 1,
    });
  });

  new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              const layerKey = context.dataset.label.split(" ")[0];
              const item = dataByLayers[layerKey][context.dataIndex];
              return `${context.dataset.label}, ${item.neurons} neurons: MSE = ${item.loss.toFixed(6)}, R² = ${item.r2.toFixed(4)}`;
            },
          },
        },
        title: {
          display: true,
          text: "Architecture Performance Comparison",
        },
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: {
            display: true,
            text: "Neurons Per Layer",
          },
        },
        y: {
          title: {
            display: true,
            text: "Test Loss (MSE)",
          },
        },
      },
    },
  });
}
