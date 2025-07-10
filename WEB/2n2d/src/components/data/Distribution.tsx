import Plot from "react-plotly.js";

export default function FeatureDistributions({distribution_data}) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {Object.entries(distribution_data).map(([feature, dist]) => (
                <div key={feature} className="bg-background rounded-xl p-4 shadow">
                    <Plot
                        data={[
                            {
                                x: dist.bins.slice(1),
                                y: dist.counts,
                                type: "bar",
                                marker: {color: "#3b82f6"},
                            },
                        ]}
                        layout={{
                            title: {
                                text: feature,
                                font: {size: 18, color: "#ffffff"},
                            },
                            paper_bgcolor: "transparent",        Testing content: ${!!testingContent}
                              const filename = document.getElementById("data-filename");
                                      layersTableBody.innerHTML = "";

                                              hierarchicalRepulsion: {
                        });
                        }
                              testSizeSlider.addEventListener("input", () => {
                                      const targetSelect = document.getElementById("target-select");
                                            document
                                      data: [
                        } else {
                        },
                                  featureCheckboxes.innerHTML = "<p>No columns available</p>";
                                    architecture.output_layer,
                                            <div class="info-row">
                                const canvas = document.getElementById("architecture-canvas");

                                                </tr>




                                    '<tr><td colspan="3">No layer information available</td></tr>';
                                                            <strong>Epochs:</strong> ${architecture.epochs}
                        }
                    <strong>Producer:</div>strong> ${summary.producer || "Unknown"}
                  const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
                          columns.forEach((column) => {

            }
                const pageData = csvData.data.slice(startIndex, endIndex);
                              const file = event.target.files[0];
                                          data: layers,
                    const byteArrays = [];
                                                  tr.appendChild(td);
                                                        window.csvData = {
                                                                populateFeatureDropdowns();
                                                                    if (summary.inputs && summary.inputs.length > 0) {
                                                                            if (loadCsvButton && csvInput) {
                                                                                  renderer.render(scene, camera);

                                                                                      columns.forEach((column) => {
                if (!summary) {
                document.getElementById("metric-samples").textContent =
            },
                labelsContainer.style.zIndex = "10";


                if (!positions[edge.from] || !positions[edge.to]) {
            }

                while (nodesToProcess.length > 0) {
            },
            }
                console.error("No data available for features");
                                                                                              details = `Kernel: ${layer.kernel_size.join("Ã—")}`;
            }
                                                                                  const preprocessButton = document.getElementById("btn-preprocess");

                                                                                  ...architecture.hidden_layers,
                      html += '<ul class="model-io-list">';

                                                                                      `;
                                                                                        renderer.setSize(width, height);

                                                                                                row.appendChild(indexCell);
                                                                                                  if (!window.graphData) {
                                                                                                      layersTableBody.innerHTML = "";
                                                                                                            datasets: [
                                                                                                                          label: function (context) {

                                                                                                                            if (results.metrics) {
                                                                                                                              if (nextButton) nextButton.disabled = currentPage >= totalPages - 1;


                                                                                                                                  const hiddenLayersStr = architecture.hidden_layers.join(", ");
                                                                                                                                      if (window.csvData) {
                                                                                                                                              currentPage = 0;
                                                                                                                                                  try {
                                                                                                                                                      const testSizeValue = document.getElementById("test-size-value");
                                                                                                                                                        scene.add(directionalLight);
                                                                                                                                                            container3d.innerHTML = "";
                                                                                                                                                                showSpinner(false);



                                                                                                                                                                        },
                                                                                                                                                                            const byteCharacters = atob(result.base64);
                                                                                                                                                                                    margin: 10,
                                                                                                                                                                                      camera = new THREE.PerspectiveCamera(40, aspect, 1, 10000);
                                                                                                                                                                                            const result = await window.eel.load_onnx_model(base64)();
                                                                                                                                                                                                }
                                                                                                                                                                                                          pointRadius: 5,

                                                                                                                                                                                                                document

                                                                                                                                                                                                                    if (!modelDetailsElement) return;
                                                                                                                                                                                                                        const layerLabels = [
                                                                                                                                                                                                                                shuffleData,
                                                                                                                                                                                                                                  function renderCsvTable(data) {


                                                                                                                                                                                                                                                  ${output.type ? `<span class="io-type">${output.type}</span>` : ""}
                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                      const fov = camera.fov * (Math.PI / 180);
                                                                                                                                                                                                                                                        });
                                                                                                                                                                                                                                                          Object.values(nodeObjects).forEach((node) => {

                                                                                                                                                                                                                                                            if (!container) {
                                                                                                                                                                                                                                                                featureCheckboxes.innerHTML = "";

                                                                                                                                                                                                                                                                            backgroundColor: [
                                                                                                                                                                                                                                                                                sphere.userData = { id: node.id, label: node.label, type: node.type };
                                                                                                                                                                                                                                                                                        const testSizeSlider = document.getElementById("test-size");
                                                                                                                                                                                                                                                                                              prevButton.disabled = currentPage === 0;
                                                                                                                                                                                                                                                                                                        borderWidth: 2,
                                                                                                                                                                                                                                                                                                            const featureCheckboxes = document.getElementById("feature-checkboxes");
                                                                                                                                                                                                                                                                                                                          label: function (context) {
                                                                                                                                                                                                                                                                                                                            const positions = {};
                                                                                                                                                                                                                                                                                                                                  },
                                                                                                                                                                                                                                                                                                                                      }
ulate
                                                                                                                                                                                                                                                                                                                                        for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                                                                                                                                                                                                                                                                                                                                                                <strong>${input.name}</strong>
                                                                                                                                                                                                                                                                                                                                                                          label: "Perfect Prediction",
                                                                                                                                                                                                                                                                                                                                                                            const nodePositions`
            }
            })
            }
            }
            })
            })
                        }
                            ]
                        })
                            plot_bgcolor: "transparent",
                            xaxis: {
                                title: "Value",
                                tickfont: {color: "#cccccc"},
                                titlefont: {color: "#cccccc"},
                            },
                            yaxis: {
                                title: "Count",
                                tickfont: {color: "#cccccc"},
                                titlefont: {color: "#cccccc"},
                            },
                            margin: {t: 40, b: 50, l: 50, r: 30},
                        }}
                        config={{responsive: true, displayModeBar: false}}
                        style={{width: "100%", height: "300px"}}
                    />
                </div>
            ))}
        </div>
    );
}