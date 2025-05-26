      const cell = document.createElement("td");
      });
    if function populateFeatureSelectors(columnsInfo) {
        const byteArray = new Uint8Array(byteNumbers);
    }

    let currentPage = 0;
            y: offsetY,
                      return `${context.raw} neurons`;
                const thead = table.querySelector("thead");
                      updateStatus(`Error: ${error.message}`);

                        const byteArrays = [];
                                updateStatus(`Error: ${pageData.error}`);
                                        typeCell.textContent = opType;
                                            updateStatus(`Error: ${error.message}`);
                                              console.log(`Model available: ${hasModel}, Data available: ${hasData}`);
                                                function displayLayersInfo(layers) {
                                                          const preprocessButton = document.getElementById("btn-preprocess");

                                                          </>ul>
                updateStatus(`Loading ${file.name}...`);

                contentPane.classList.add("active");
                          borderWidth: 1,
                                  const formatNumber = (  const nodeSpacing = 120;
                                        updateStatus(`Error: ${error.message}`);
                                                    row.appendChild(nameCell);
                                                                        <div class={"detail-item">
                                                                        }num) => (Number.isFinite(num) ? num.toFixed(4) : "-");
                                row.appendChild(cell);
                                  try {
                                          const layerLabels = [
                                                            "rgba(54, 162, 235, 0.7)",
                                                    testSizeSlider.addEventListener("input", () => {
                                                        renderer.domElement.style.top = "40px";
                                                        showSpinner(true);
                                                        console.log("Waiting for Eel to initialize...");
                                                        updateStatus("Running model test...");
                                                        showSpinner(true);
                                                        const pageSize = 10;
                                                        container2d.style.display = "none";

                                                    }
                                                            tooltip: {
                                                                  data: results.map((item) => ({ x: item.actual, y: item.predicted })),
                                                                          if (runTestButton) {
                                                                                const testSize = parseFloat(testSizeSlider.value);
                                                                                        clearDataButton.disabled = false;
                                                                                          nodes.forEach((node) => {
                                                                                              <h4>Outputs</h4>
                                                                                              th.textContent = column;
                                                                                              totalRow.appendChild(totalCountCell);


                                                                                          } catch (e) {

                                                                                                    span.textContent = column.name;

                                                                                                        if (typeof window.eel === "undefined") {
                                                                                                                                </div>
                                                                                                              setTimeout(checkTabVisibility, 1000);
                                                                                                            (option) => option.value,
                                                                                                                      console.log("Waiting for Eel to initialize...");
                                                                                                                                <tr>
                                                                                                                                          totalRow.appendChild(totalLabelCell);
                                                                                                                                    });
                                                                                                                                    function displayModelSummary(summary) {
                                                                                                                                      if (filename) filename.textContent = summary.filename || "Unnamed dataset";
    const tbody = table.querySelector("tbody");
        const layersTableBody = document.querySelector("#layers-table tbody");
            const thead = table.querySelector("thead");

                                                                                                                                    )();
                                                                                                const option = document.createElement("option");
                                                                                                                                }

                                                                                                                                      startOptimizationButton.addEventListener("click", startOptimization);
                                                                                                                                            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
                                                                                                                                                <strong>IR Version:</strong> ${summary.ir_version || "Unknown"}

                                                                                                                                </tr>div>

                                                                                                                  modelDetailsElement.innerHTML = `
                                                                                                                      });
                                                                                                                      
                                                                                                                      
                                                                                                                      
                                                                                                                          const downloadModelButton = document.getElementById("download-model");
                                                                                                                              if (startOptimizationButton) {
                                                                                                                                  option.value = column;
                                                                                                                                      const pos = positions[node.id];
                                                                                                                                      
                                                                                                                                            targetSelect.appendChild(option);
                                                                                                                                                  });
                                                                                                                                                      } catch (error) {
                                                                                                                                                                      </div>
      updateOptimizationUI();

    const ctx = canvas.getContext("2d");
          result.summary.nodes.forEach((node, index) => {
        updateDataSummary({

  function updateModelDetails(summary) {
          text: "Predicted vs Actual Values",
      console.error("Error loading CSV:", error);
    if (prevPageButton) {
  controls.maxDistance = 5000;
    });
      pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
                      <div class="architecture-details">
    animationFrameId = null;
                        <th>Domain</th>
                              const result = await window.eel.preprocess_data(
                                  });
                                  
                                      }
                                                iterations: 1000,
                                                  if (view2dButton) view2dButton.classList.toggle("active", mode === "2d");
                                                  
                                                          const opType = node.op_type || "Unknown";
                                                              container3d.style.width = "100%";
                                                                      datasets: [
                                                                          showSpinner(true);
                                                                              }
                                                                                        currentPage--;
                                                                                        
                                                                                              nextPageButton.addEventListener("click", () => {
                                                                                                updateOptimizationUI();
                                                                                                
                                                                                                    updateStatus("Load a model first");
                                                                                                        const tbody = table.querySelector("tbody");
                                                                                                              node.layer = 0;
                                                                                                                      currentPage = 0;
                                                                                                                          }
                                                                                                                          </th>`</strong>
                                                                                                                                }
                                                                                                                                </tr>
                                                                                                        }
                                                                              })</h4>
                                                                                          })
                                                                          }
                                                    })
                                          ]
                                  }    });

                    ${outputShapesHtml || "<p>No output information available</p>"}
      }
      }
      }
      et graphData =  (tableBody)
      {
          const inputFeatures = document.getElementById("opt-input-features");

      } else {
      updatePaginationControls(page, totalPages);
            ...architecture.hidden_layers.map((_, i) => `Hidden ${i + 1}`),

                                  </div>

                      </li>`;
                              <div class="detail-section">
    } else {
  container.style.position = "relative";

          pointHoverRadius: 7,

  if (nextButton) nextButton.disabled = currentPage >= totalPages - 1;
      return;
  if (results.results && results.results.length > 0) {

                    </div>li>
`
      }
            const indexCell = documenteElement("td");
  if (modelInfoContainer) {
  }
        const span = document.createElement("span");


      }
            downloadModelButton.addEventListener("click", downloadOptimizedModel);
function setupThreeJsScene(con        '<div class="error-message">No valid graph data available</div>';
        <div class={"detail-item">    createPredictionChart(results.results);

        }
                 if (!runTestButton) return;
  document.querySelectorAll(".tab-pane").forEach((pane) => {
        const detailsElement = document.getElementById("model-details");

  )
      {
          const size = node.size || 15;
          try {
              incomingEdges[edge.to]++;
              arrows: {
                  to: {
                      enabled: true, scaleFactor
                  :
                      0.5
                  }
              }
          ,
              html += "</ul>";
              layersTableBody.innerHTML = "";


              <h4>Model Information</h4>

              layersTableBody.innerHTML = "";

          } catch (error) {
              <strong>Version:</strong>
              $
              {
                  modelInfo.producer_version || "Unknown"
              }

              const columns = Object.keys(data[0]);
              architecture.input_layer,
          }
      }
        scales: {
            console.log("Download model button event listener added");
            const min = Math.min(...allValues);
            borderWidth: 1,

                let
            currentLayer = 0;
            maintainAspectRatio: false,
        }
        const cell = document.createElement("td");

  function updateStatus(message) {
  }
  });
                            </div>
      });
      }
                    <div class={"param-info">
                                        </tr>
                        const nextButton = document.getElementById("next-page");

                          const base64 = arrayBufferToBase64(arrayBuffer);
                          modelInfoElement.innerHTML = `
                            const height = container.clientHeight || 600;
                                        <strong>Producer:</strong> ${summary.producer || "Unknown"}
                                            const material = new THREE.MeshLambertMaterial({
                                              labelsContainer.style.top = "0";

                                                  html += '<div class="detail-section"><h4>Outputs</h4>';
                                                        let outputShapesHtml = "";
                                                              console.log("Start optimization button event listener added");
                                                                      details = `Kernel: ${layer.kernel_size.join("Ã—")}`;

                                                                        }
                                                                            const result = await window.eel.download_optimized_model(

                                                                            }

                                                                                updateStatus(`Error: ${error.message}`);
                                                                                    if (!data || data.length === 0) {


                                                                                            updateStatus("Data cleared");
                                                                                                      text: "Predicted vs Actual Values",

                                                                                                              <div class="detail-section">
                                                                                                                  console.log("Showing empty state message");
                                                                                                                      } else {
                                                                                                                            });

                                                                                                                              });
                                                                                                                                  const container = document.getElementById("architecture-display");
                                                                                                                                        const offsetY = (i - (nodesInLayer.length - 1) / 2) * nodeSpacing;
                                                                                                                                              html += "<tbody>";
                                                                                                                                                    type: "bar",
                                                                                                                                                      }
                                                                                                                                                      });
                                                                                                                                                            };
                                                                                                                                                                } catch (error) {
                                                                                                                                                                        const td = document.createElement("td");
                                                                                                                                                                          const targetFeatureSelect = document.getElementById("target-feature");
                                                                                                                                                                                              </div>
                                                                                                                                                                                                document.querySelectorAll(".tab-pane").forEach((pane) => {
                                                                                                                                                                                                            ],

                                                                                                                                                                                                                const thead = table.querySelector("thead");
                                                                                                                                                                                                                      reader.onerror = (e) => reject(new Error("File reading failed"));
                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                updateDataPreview(result.data);
                                                                                                                                                                                                                                        <div class="detail-item">
                                                                                                                                                                                                                                            0x8b5cf6,
                                                                                                                                                                                                                                                });
                                                                                                                                                                                                                                                      headerRow.appendChild(th);
                                                                                                                                                                                                                                                            aspectRatio: 1.5,


                                                                                                                                                                                                                                                                modelInfoContainer.innerHTML = `
                        featureCheckboxes.innerHTML = "";

                        tabButton.click();
                                        <div class="param-info">

                                        function displayModelSummary(summary) {
                                                        </div>
                                    display: true,
                            document.getElementById("metric-mse").textContent = formatNumber(
                                            text: "Actual Values",
                          const labelsContainer = document.getElementById("three-labels-container");
                                                labelDiv.textContent = text;

                                                      let inputShapesHtml = "";

                    },

                label: "Perfect Prediction",
                      controls.zoomSpeed = 1.0;

                        )
                    }}</strong></h4>
              }
  })
                </div>
              runTestButton.addEventListener("click", () => {
                        updateStatus(`Error: ${result.error}`);
                              tbody.innerHTML = '<tr><td colspan="100">No data available</td></tr>';
                                    modelDetailsElement.innerHTML = `
                                    `
        })})
  }  const emptyState = document.getElementById("data-empty-state");


      }
          const x = ((worldPos.x + 1) / 2) * rendererRect.width;

      let outputShapesHtml = "";
                      <strong>${input.name || "Unnamed"}</strong>
  const status = document.getElementById("status");
          fill: false,



      }

    container2d.classList.remove("active");
  const nodePositions = calculateNodePositions(data.nodes, data.edges);
        clearDataButton.classList.remove("disabled");
      }
      }
    console.error("Error creating 2D network:", error);

    a.href = url;
      option.value = column.name;
      tr.appendChild(detailsCell);
                "rgba(75, 192, 192, 1)",
          const edges = new vis.DataSet(graphData.edges);
    tabButton.click();
      }


    createPredictionChart(results.results);
      label.style.display = "block";
      architecture.input_layer,
        labelsContainer.style.top = "0";
        outputsHtml += `
                Empty state: ${!!emptyState}
                
                      });
                          const indexHeader = document.createElement("th");
                            } else {
                                  displayModelInfo(summary.model_info);
                                        const preprocessButton = document.getElementById("btn-preprocess");
                                              headerRow.appendChild(th);
                                                        {
                                                        
                                                              });
                                                                    cell.colSpan = 3;
                                                                                <h3>Recommended Architecture</h3>
      nextPageButton.addEventListener("click", () => {

    }
    let inputsHtml = "<h4>Inputs</h4>";
            <h4>Model Information</h4>
                setTimeout(() => {
                
                    if (runTestButton) runTestButton.disabled = true;
                        const columns = Object.keys(data[0]);
                        
                                    <div class="architecture-card">
                                          html += '<table class="layer-types-table">';
                                          
                                            nodes.forEach((node) => {
                                              function displayModelInfo(modelInfo) {
                                                      summary: result.summary,
                                                            tableBody.appendChild(tr);
                                                                    const emptyState = document.getElementById("data-empty-state");
                                                                    
                                                                      container.style.position = "relative";
                                                                                display: true,
                                                                                      return;
                                                                                      
                                                                                              },
                                                                                                      .classList.remove("hidden");
                                                                                                              return;
                                                                                                              
                                                                                                                const allValues = [...actual, ...predicted];
                                                                                                                    return;
                                                                                                                      });
                                                                                                                              },
                                                                                                                              
                                                                                                                                if (tableView) tableView.style.display = "block";
                                                                                                                                }
                                                                                                                                                    <div class="layer-info">
                                                                                                                                                        `;
          distance *= 1.5;


          function createPredictionChart(results) {

                  try {
                      const layerKeys = Object.keys(layers)

                      async function downloadOptimizedModel() {
                      }

                  ,
                      if (modelInfoElement) {
                      }


                      currentLayer++;
                  } finally {
                  }
                    console.log(`Switching to tab: ${tabId}`);
                          </div>
                if (nextPageButton) {


                    <strong>Epochs:</strong>
                    $
                    {
                        architecture.epochs
                    }
                } else {
                } else {
                    nodeObjects[node.id] = sphere;
                    return btoa(binary);
              ) {
      layersTableBody.innerHTML =

    label.style.top = `${y}px`;
    setTimeout(() => {
                console.error("Error generating summary:", summary.error);

                      showSpinner(true);
                      function displayArchitectureSuggestions(result) {

                              const hiddenLayersStr = architecture.hidden_layers.join(", ");
                      );
        container2d.style.display = "none";
                      }
                            console.log("Network stabilized");
                              node.layer = currentLayer;
                                      details = `Shape: [${layer.shape.join(", ")}]`;
    }

      createEdges(data.edges, nodePositions);
                <ul>
                      const pageInfo = document.getElementById("page-info");
                    {
                              const detailsCell = document.createElement("td");
                    }
                          showSpinner(false);
                                                <span class={"detail-type">${input.data_type}</span>
                                                    const formatNumber = (num) => (Number.isFinite(num) ? num.toFixed(4) : "-");
                                                },


                  document.body.removeChild(a);

                      const tr = document.createElement("tr");
                      type: "bar",
                          <h4>Inputs (${summary.inputs?.length || 0})</h4>
                      loadCsvPage();
                document.addEventListener("DOMContentLoaded", function () {
                    <p><strong>IR Version:</strong> ${summary.ir_version || "Unknown"}</p>p>
                                ],

                } finally {
                  async function preprocessData() {

                  }
                        label.className = "feature-checkbox";
                                      </li>
                  });
                                    <h4>Model Information</h4>
        network.fit();
    const prevPageButton = document.getElementById("prev-page");


    nodesInLayer.forEach((node, i) => {
                'input[name="complexity"]:checked',
                    <p><strong>Learning Rate:</strong> ${result.learning_rate}</p>p>


                row.appendChild(indexCell);
      const colors = [
    }
        if (!nodeTypes[type]) {
            totalCountCell.textContent = `${summary.nodes.length} nodes`;

            <h3>Recommended Network Structure</h3>
            displayCsvData(window.csvData, currentPage);
            document.getElementById("metric-samples").textContent =
                labelObjects[node.id] = labelDiv;
            nodesToProcess.push(...nextLayerNodes);
        },
                  container.innerHTML = "";
    const thead = table.querySelector("thead");

      results.metrics.mae,
                  const indexHeader = document.createElement("th");

                  {

                                return `Actual: ${context.parsed.x.toFixed(4)}, Predicted: ${context.parsed.y.toFixed(4)}`;

                    if (outputsContainer) {
                        labelsContainer.id = "three-labels-container";
                        targetOption.textContent = column;


                    }
                      const labelsContainer = document.getElementById("three-labels-container");
                      Object.keys(nodeObjects).forEach((nodeId) => {
                      )();
                            loadCsvButton.addEventListener("click", () => {

                                if (columnCount) columnCount.textContent = summary.columns || "-";

                                plugins: {
                                    <strong>Input Layer:</strong>
                                    $
                                    {
                                        architecture.input_layer
                                    }
                                    neurons
                                    if (runTestButton) runTestButton.disabled = false;
                                    !window.csvData ||
                                }
                            ,
                                <p><strong>Batch Size:</strong> ${result.batch_size}</p>
                                p >
                                const aspect = width / height;
                            }
                            summary.nodes.forEach((node) => {
                                    if (typeof window.eel === "undefined") {
                                    } catch (error) {

                                            scene.add(sphere);
                                              scene.background = new THREE.Color(0xffffff);

                                                  document.getElementById("metric-mae").textContent = formatNumber(
                                                            updateStatus(`Error: ${error.message}`);

                                }
                                            updateStatus("Data cleared");
                                        if (window.graphData) {
                                              const filename = document.getElementById("data-filename");
                                                              <td>${count} node${count > 1 ? "s" : ""}</td>
                                        let network = null;
                                                        <table class={"info-table">
                                                            });
                                        },
                                      opacity: 0.9,
                                                        ${output.type ? `<span class="io-type">${output.type}</span>` : ""}
                                            const hasFeatures = featureCheckboxes.length > 0;
                                                filename: file.name,


                                                        <strong>Output Layer:</strong> ${architecture.output_layer} neurons
                                        const modelDetailsElement = document.getElementById("model-details");
                            },

                                  tr.appendChild(td);

                                label.style.transform = "translate(-50%, -50%)";
                                nodesInLayer.forEach((node, i) => {
                                          reader.readAsArrayBuffer(file);
                                              nodeObjects[node.id] = sphere;
                                                  cancelAnimationFrame(animationFrameId);
                                                      try {

                                                      },
                                        status.textContent = message;
                                                            html += '<ul class="model-io-list">';
                                                                      ],
                                        labelDiv.className = "node-label";

                          <div class={"param-info">
                                              <span class="info-value">${summary.producer || "Unknown"}</span>
                                  "display: block !important;";
                          }
                                      'input[name="complexity"]:checked',
                                  nodes.forEach((node) => {
                                      if (preprocessButton) preprocessButton.disabled = true;
                                      tr.appendChild(td);

                                      <h4>Model Information</h4>

                                      showError(`Postprocessing failed: ${error.message}`);

                                      edges.forEach((edge) => {
                                      });
                                      typeCell.textContent = layer.type || "Unknown";

                                  }
                                      labelObjects = {};

                                        network = null;
                                            const tabButtons = document.querySelectorAll(".tab-button");
                                                const featureCheckboxes = document.getElementById("feature-checkboxes");
                                } else if (layer.type === "Linear" && layer.shape) {
                              results.metrics.samples;
                              return;
                              let details = "";
                          }

                              if (!columnsInfo || columnsInfo.length === 0) {
                                    layerKeys.forEach((layerIndex) => {
                                            sphere.userData = { id: node.id, label: node.label, type: node.type };
                                              nodes.forEach((node) => {
                                                    const headerRow = document.getElementById("csv-headers");
                                                          edges: {
                                                              function updatePaginationControls(currentPage, totalPages) {
                                                                    if (modelInfoContainer) {
                                                                        renderer.domElement.style.top = "40px";
                                                                        <h4>Outputs
                                                                            (${summary.outputs?.length || 0})</h4>
                                                                    }
                                                                const hasData =
                                                                  console.log(`Showing tab: ${contentId}`);
                                                                  0x4f46e5, 0x7c3aed, 0xe11d48, 0xf59e0b, 0x10b981, 0x3b82f6, 0xec4899,
                                                                    th.textContent = column;
                                                                              <table class={"info-table">
                                                                                  if (!table) return;
                                                                                  updateStatus("Preparing model for download...");
                                                                                          <td><strong>Total</strong></td>
                                                                                    tr.appendChild(detailsCell);
                                                                                      summary.outputs.forEach((output) => {
                                                                                  <p><strong>Domain:</strong> ${summary.domain || "General"}
                                                              </p>
                                                                  modelInfoElement.innerHTML = `
                                                                                                }
                                                                                                
                                                                                                    const line = new THREE.Line(geometry, material);
                                                                                                                        </div>
}
        datasets: [
      incomingEdges[edge.to]++;

  inputFeatures.setAttribute("multiple", "multiple");
        summary.inputs.forEach((input) => {
        stabilization: {
        const tableView = document.getElementById("data-table-view");



    const indexHeader = document.createElement("th");

            nameCell.textContent = node.name || "Unnamed";
    }

          ],
  const canvasElements = container.querySelectorAll("canvas");
    if (!nodeTypes[type]) {
    cancelAnimationFrame(animationFrameId);
    const line = new THREE.Line(geometry, material);
    const line = new THREE.Line(geometry, material);
      startOptimizationButton.addEventListener("click", startOptimization);
    const slice = byteCharacters.slice(offset, offset + 512);
    const clearDataButton = document.getElementById("btn-clear-data");
    console.log(`
                                                                  Hidden
                                                                  tab: $
                                                                  {
                                                                      pane.id
                                                                  }
                                                                  `);
      } finally {
        }
            inputFeatures.appendChild(inputOption);
                  html += "<p>No input information available</p>";
      tr.appendChild(nameCell);

  const activeButton = document.getElementById(tabId);
      console.warn(`
                                                                  Missing
                                                                  positions
                                                                  for edge $
                                                                  {
                                                                      edge.from
                                                                  }
                                                              ->
                                                                  $
                                                                  {
                                                                      edge.to
                                                                  }
                                                                  `);
      
          if (window.csvData) {
              if (!network) {
                const featureContainer = document.getElementById("feature-checkboxes");
                
                    console.error("No data available for features");
                          Object.entries(opTypeCounts).forEach(([opType, count], index) => {
                              label.style.top = `
                                                                  $
                                                                  {
                                                                      y
                                                                  }
                                                                  px`;
                              
                              
                              
                                console.log(`
                                                                  Switching
                                                                  to
                                                                  tab: $
                                                                  {
                                                                      tabId
                                                                  }
                                                                  `);
                                        assignedNodes.add(nodeId);
                                        
                                            } finally {
                                                }
                                                    pageInfo.textContent = `
                                                                  Page
                                                                  $
                                                                  {
                                                                      currentPage + 1
                                                                  }
                                                                  of
                                                                  $
                                                                  {
                                                                      totalPages
                                                                  }
                                                                  `;
                                                      } catch (error) {
                                                      function populateOptimizationFeatures() {
                                                                {
                                                                    } catch (error) {
                                                                            csvInput.click();
                                                                                  prevButton.disabled = currentPage === 0;
                                                                                              console.log(`
                                                                  Moving
                                                                  to
                                                                  page
                                                                  $
                                                                  {
                                                                      currentPage + 1
                                                                  }
                                                                  `);
                                                                                                  targetSelect.disabled = false;
                                                                                                        const th = document.createElement("th");
                                                                                                            if (summary.nodes && summary.nodes.length > 0) {
                                                                                                                      result.summary.nodes.forEach((node, index) => {
                                                                                                                      
                                                                                                                          layers[node.layer].push(node);
                                                                                                                          
                                                                                                                              }
                                                                                                                                  });
                                                                                                                                    const pageSize = 10;
                                                                                                                                          scales: {
                                                                                                                                              const preprocessButton = document.getElementById("btn-preprocess");
                                                                                                                                                  const geometry = new THREE.SphereGeometry(size, 16, 16);
                                                                                                                                                      }
                                                                                                                                                          });
                                                                                                                                                                }
                                                                                                                                                                    }
                                                                                                                                                                              const totalPages = Math.ceil(window.csvData.data.length / pageSize);
                                                                                                                                                                                          <td><strong>${totalNodes} node${totalNodes > 1 ? "s" : ""}</strong></td>td>
                                                                                                                                                                                                  const shape = output.shape ? `[$
                                                                  {
                                                                      output.shape.join(", ")
                                                                  }
                                                              ]
                                                                  ` : "";
                         <strong>Output Layer:</strong> ${architecture.output_layer} neurons
                           if (labelsContainer) {
                             Object.keys(nodeObjects).forEach((nodeId) => {
                             
                                   });
                                   }
                                   
                                   
                                       decode_predictions: document.getElementById("decode-predictions").checked,
                                         }
                                         
                                         
                                         
                                               });
                                                       legend: {
                                                         renderer.setSize(width, height);
                                                           const testSize = parseFloat(testSizeSlider.value);
                                                               type: "scatter",
                                                                   const clearDataButton = document.getElementById("btn-clear-data");
                                                                     }
                                                                     let architectureChart = null;
                                                                     
                                                                           setTimeout(() => {
                                                                             const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                                                                             
                                                                                 nextPageButton.disabled = currentPage >= totalPages - 1;
                                                                                         enabled: true,
                                                                                               setTimeout(() => {
                                                                                               function displayArchitectureSuggestions(result) {
                                                                                               
                                                                                               
                                                                                                 if (mode === "2d") {
                                                                                                     try {
                                                                                                         if (summary.nodes && summary.nodes.length > 0) {
                                                                                                               console.error("Error getting suggestions:", error);
                                                                                                                   updateDataPreview(result.data);
                                                                                                                   
                                                                                                                         showSpinner(true);
                                                                                                                                         <div class="architecture-details">
                                                                                                                                                     backgroundColor: [
                                                                                                                                                               currentPage--;
                                                                                                                                                                 const detailsElement = document.getElementById("model-details");
                                                                                                                                                                   if (!container) return;
                                                                                                                                                                         });
                                                                                                                                                                             }
                                                                                                                                                                                 const material = new THREE.MeshLambertMaterial({
                                                                                                                                                                                   const tableBody = document.getElementById("csv-body");
                                                                                                                                                                                     console.log(`
                                                                  Updating
                                                                  pagination: page
                                                                  $
                                                                  {
                                                                      currentPage + 1
                                                                  }
                                                                  of
                                                                  $
                                                                  {
                                                                      totalPages
                                                                  }
                                                                  `);
                                                                                                                                                                                         if (view3dButton) {
                                                                                                                                                                                         let graphData = null;
                                                                                                                                                                                         
                                                                                                                                                                                           labelObjects = {};
                                                                                                                                                                                               console.log("Showing empty state message");
                                                                                                                                                                                                     updatePaginationInfo();
                                                                                                                                                                                                         if (!graphData || !graphData.nodes || !graphData.edges) {
                                                                                                                                                                                                         
                                                                                                                                                                                                             const modelInput = document.getElementById("model-input");
                                                                                                                                                                                                                 const label = labelObjects[nodeId];
                                                                                                                                                                                                                   } finally {
                                                                                                                                                                                                                           const opType = node.op_type || "Unknown";
                                                                                                                                                                                                                             console.log("DOM loaded, initializing application...");
                                                                                                                                                                                                                               const aspect = width / height;
                                                                                                                                                                                                                                       },
                                                                                                                                                                                                                                               enabled: true,
                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                 }
                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                       if (clearDataButton) {
                                                                                                                                                                                                                                                               const indexCell = document.createElement("td");
                                                                                                                                                                                                                                                                     event.target.value = "";
                                                                                                                                                                                                                                                                           const row = document.createElement("tr");
                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                           }
                                                                                                                                                                                                                                                                                   missingValues,
                                                                                                                                                                                                                                                                                         console.log("Waiting for Eel to initialize...");
                                                                                                                                                                                                                                                                                                 updateStatus(`
                                                                  Error: $
                                                                  {
                                                                      pageData.error
                                                                  }
                                                                  `);
                                                                                                                                                                                                                                                                                                     if (!container) return;
                                                                                                                                                                                                                                                                                                               displayCsvData(window.csvData, currentPage);
                                                                                                                                                                                                                                                                                                                 console.log(`
                                                                  Switching
                                                                  to
                                                                  $
                                                                  {
                                                                      mode
                                                                  }
                                                                  mode`);
                                                                                                                                                                                                                                                                                                                     const inputOption = document.createElement("option");
                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                         }
                                                                                                                                                                                                                                                                                                                           const colors = [
                                                                                                                                                                                                                                                                                                                               try {
                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                       <div class="detail-section">
      showSpinner(true);
        network.redraw();
  container.appendChild(renderer.domElement);

        enabled: true,
      featureCheckboxes.innerHTML = "<p>No columns available</p>";
        scene.background = new THREE.Color(0xffffff);
                  label: "Predictions",
                  function displayCsvData(csvData, page = 0) {
                  
                  function populateFeatureSelectors(columns) {
                    targetFeature.innerHTML =
                      const resultsContainer = document.getElementById("testing-results");
                      
                        controls.enableDamping = true;
                          const testSize = parseFloat(testSizeSlider.value);
                                const pageData = await window.eel.get_csv_page(currentPage, pageSize)();
                                        shadow: true,
                                        
                                                    <div class="architecture-card">
                                                        const options = {
                                                            }
                                                            }
                                                                    </div>
                                                                      } catch (error) {
                        th.textContent = column;
                      container2d.classList.remove("active");
                    });

            <p><strong>Domain:</strong> ${summary.domain || "General"}</p>
                  });
                    };
                            handleTabChange(tabId);
                              controls.rotateSpeed = 0.5;
                              
                                  updateOptimizationUI();
                                            springLength: 120,
                                                          label: function (context) {
                                                                                                                                                                                                                                                                                                                                     if (inputFeatures.length === 0) {
                                                                                                                                                                                                                                                                                                                                         const layersTableBody = document.querySelector("#layers-table tbody");
                                                                                                                                                                                                                                                                                                                                           binary += String.fromCharCode(uint8Array[i]);
                                                                                                                                                                                                                                                                                                                                                             <strong>Producer:</strong> ${modelInfo.producer || "Unknown"}
                                                                                                                                                                                                                                                                                                                                                                     },
                                                                                                                                                                                                                                                                                                                                                                                 { x: max, y: max },
                                                                                                                                                                                                                                                                                                                                                                                     console.log("Showing testing UI and populating feature dropdowns");
                                                                                                                                                                                                                                                                                                                                             const opType = node.op_type || "Unknown";
                                                                                                                                                                                                                                                                                                                                                       });
                                                                                                                                                                                                                                                                                                                                                         animationFrameId = requestAnimationFrame(animate);
                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                             for (let i = 0; i < slice.length; i++) {
                                                                                                                                                                                                                                                                                                                                                                       testSize,
                                                                                                                                                                                                                                                                                                                                                                         if (!inputFeatures || !targetFeature) {
                                                                                                                                                                                                                                                                                                                                                                               });
      const indexCell = document.createElement("td");
      
        if (controls) controls.update();
              });
                  layersTableBody.innerHTML = "";
                    targetFeature.innerHTML =
                        });
                                font: { size: 14, face: "Inter" },
                                    const file = event.target.files[0];
                                    
                                        const nodeMesh = nodeObjects[nodeId];
                                        
                                                    ],
                                                                display: true,
                                                                  scene.add(ambientLight);
                                                                          font: { size: 14, face: "Inter" },
                                                                                  td.dataset.row = index;
                                                                                        labelObjects = {};
                                                                                            document.getElementById("metric-r2").textContent = formatNumber(
                                                                                                        row.appendChild(typeCell);
                                                                                                            }
                                                                                                            
                                                                                                              }
                                                                                                                    scales: {
          }
            });
                status.textContent = message;
                    }
                                        <div class={"param-info">
                                        
                                              return;
                                            const pageData = csvData.data.slice(startIndex, endIndex);
                                                layersTableBody.appendChild(row);
                                                            <div class="param-info">
                                                                if (!nodeTypes[type]) {
                                                                
                                                                    }
                                                                          results.metrics.r2,
                                                                          
                                                                            console.log("Updating model testing UI...");
                                                                                  }
                                                                                  
                                                                                  
                                                                                        return;
                                                                                          if (emptyState) emptyState.style.display = "none";
                                                                                                              <span class={"info-value">${summary.domain || "General"}</span>
                                                                                                                    },
                                                                                                                    
                                                                                                                        </div>`;

                                                              },
                                                                        if (pageData.error) {
                                                                          }
                                                                          scene = new THREE.Scene();
                                                              }
                                                                    const th = document.createElement("th");
                                                                                            tr.appendChild(detailsCell);
                                                                                              if (!csvData || !csvData.data || csvData.data.length === 0) {

                                                                                                  const cell = document.createElement("td");
                                                                                                  <strong>${output.name || "Unnamed"}</strong>

                                                                                              }
                                                                                          nameCell.textContent = node.name || "Unnamed";

                                                              );
                                                                                          backgroundColor: [

                                                                                                  console.log("No graph data available");
                                                              },
                                                                    ],
                                                                  console.log("Setting up event listeners...");
                                                                                            console.log("DOM loaded, initializing application...");
                                                                                                if (!data || data.length === 0) {
                                                                                                      }
                                                          }


                                                                  const td = document.createElement("td");

                                                                            "rgba(54, 162, 235, 1)",
                                                                                            <h4>Basic Info</h4>

                                                            scene.add(ambientLight);
                                              }
                                                  if (worldPos.z > 1 || worldPos.z < -1) {
                                                  },
                                                        <span class={"info-value">${summary.node_count || summary.nodes?.length || 0}</span>
                                                                                <div class="detail-item">
                                                          const nextButton = document.getElementById("next-page");

                                                                                <strong>${output.name}</strong>
                                                          const layerKeys = Object.keys(layers)
                                                              view2dButton.addEventListener("click", () => {
                                                                  if (window.csvData) {
                                                                    } catch (error) {
                                                                            const indexCell = document.createElement("td");
                                                                                        label: function (context) {
                                                                                                  outputsHtml += "</ul>";
                                                                                                          clearDataButton.disabled = false;
                                                                                                            function displayLayersInfo(layers) {
                                                                                                                  container.style.minHeight = "400px";

                                                                                                                      labelsContainer.remove();
                                                                                                                      function populateOptimizationFeatures() {
                                                                                                                          html += '<ul class="model-io-list">';
                                                                                                                          rows: "-",
                                                                                                                              updateOptimizationUI();
                                                                                                                          renderCsvTable(pageData.data);
                                                                                                                          headerRow.appendChild(indexHeader);


                                                                                                                      }

                                                                                                                              scales: {
                                                                                                                              }
                                                                                                                                    return;
                                                                                                                            console.log("Final visibility check completed");
                                                                                                            )();
                                                                return;
                                                                                                            }
                                                                                        }

                                                                                        let threeJsRenderer = null;
                                                                                          const predicted = results.map((item) => item.predicted);
                                                                                              network.once("stabilizationIterationsDone", () => {

                                                                                                  row.appendChild(indexCell);
                                                                                                  console.error(result.error);
                                                                                                  <td>
                                                                                                      <strong>${totalNodes} node${totalNodes > 1 ? "s" : ""}</strong>
                                                                                                  </td>
                                                                                                  td >

                                                                                                  td.textContent = row[column] !== null ? row[column] : "";
                                                                                              };
                                                                        summary.inputs.forEach((input) => {
                                                                            while (inputFeatures.options.length > 0) {
                                                                            }

                                                                            console.error("Error loading page:", error);


                                                                            checkbox.value = column.name;

                                                                        }
                                                                                    const nameCell = document.createElement("td");

                                                                                  filename: "No file loaded",

                                                                  }
                                                                        headerRow.appendChild(th);
                                                              }

                                    }

                                                      const hiddenLayersStr = architecture.hidden_layers.join(", ");
                                                        if (result.error) {
                                                                  ...architecture.hidden_layers,
                                                                        </div>
                                                                updateStatus("Load a model first");

                                                                async function downloadOptimizedModel() {
                                                                          return;
                                                                                  </div>
                                                                          csvInput.addEventListener("change", handleCsvUpload);
                                                                }
                                                                    container2d.classList.add("active");
                                                        }

                                                          function updatePaginationInfo() {
                                            pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
                                                  emissiveIntensity: 0.2,
                                                      <td><strong>${totalNodes} node${totalNodes > 1 ? "s" : ""}</strong></td>td>
                                                                      ],
                                                            detailsElement.innerHTML = html;
                                      if (rowCount) rowCount.textContent = summary.rows || "-";
                                        console.log("Event listeners set up successfully");
                                                          }



                                const totalPages = Math.ceil(csvData.data.length / pageSize);
                              }


                      }
                  } catch (error) {
                        container.appendChild(renderer.domElement);

                            for (let i = 0; i < slice.length; i++) {
                                inputFeatures.setAttribute("size", "5");
                                nodeMap[node.id] = node;

                                `Creating 2D visualization with ${graphData.nodes.length} nodes`,

                            }
                          if (!data || data.length === 0) {

                              inputFeatures.remove(0);
                            if (hasModel && hasData) {
                                      Object.entries(opTypeCounts).forEach(([opType, count]) => {
                                          span.textContent = column.name;

                                      } else if (layer.type === "Linear" && layer.shape) {
                                ).value;
                                          if (summary.inputs && summary.inputs.length > 0) {

                                          }
                                                              </tr>
                                        window.csvData && window.csvData.data && window.csvData.data.length > 0;
                                              label.className = "feature-checkbox";
                                }
                            function create3DNetworkVisualization(data) {
                                        title: {

                                            updateStatus("Preparing model for download...");
                                            updateModelTestingUI();
                                        },
                                      summary.nodes.forEach((node) => {
                                          nodeObjects = {};
                                      };
                                      function updateModelTestingUI() {


                                                if (result.error) {
                                                      const nodeMap = {};
                                                              row.appendChild(indexCell);

                                                                if (!targetFeature) {
                                                                        tabButtons.forEach((button) => {
                                                                                if (loadCsvButton && csvInput) {
                                                                                          button.addEventListener("click", () => {
                                                                                                  if (modelInfoElement) {
                                                                                                                          <span class={"info-value">${summary.producer || "Unknown"}</span>
                                                                                                                            let labelsContainer = document.getElementById("three-labels-container");
                                                                                                                                      <div class="details">


                                                                                                                          },
                                                                                                                              document.getElementById("metric-mse").textContent = formatNumber(

                                                                                                                                  `;
                                                                                                                              });
                                                                                                        totalCountCell.style.fontWeight = "bold";
                                                                                                                          <strong>Version:</strong>strong> ${modelInfo.producer_version || "Unknown"}
                                                                                                                                    layersTableBody.innerHTML =
                                                                                                                              } finally {
                                                                                                      function updateStatus(message) {
                                                                                                              ];
        const opType = node.op_type || "Unknown";
            if (window.graphData) {
            } else {
                try {
                    const preprocessButton = document.getElementById("btn-preprocess");
                }

            }
                let binary = "";
                const label = document.createElement("label");
                                                                                                      }


                                                                                                        if (pageInfo) {

                                                                                                                              console.log(`Model available: ${hasModel}, Data available: ${hasData}`);
                                                                                                                          }
                                                                                                          if (hasModel && hasData) {



                                                                                                                  featureContainer.appendChild(label);
                                                                                                                      if (!layers || layers.length === 0) {
                                                                                                                          });
                                                                                                renderer = null;
                                                                                                                          }
                                                                                                                  console.log("Previous page button clicked");
                                                                                                              const ctx = canvas.getContext("2d");
                                                                                                                const layerSpacing = 200;
                                                                                                                          } else {
                                                                                                                          <h4>Model Information</h4>
                                                                                                        const th = document.createElement("th");
                                                                                                                          }
                                                                                                csvData.rowCount = result.rows;
                                                                                                const clearDataButton = document.getElementById("btn-clear-data");

                                                                                            if (controls) controls.update();
                                                                                              console.error("Feature dropdown elements not found");
                                                                                            if (labelsContainer) {
                                                                                                          if (summary.inputs && summary.inputs.length > 0) {

                                                                                                                              cell.colSpan = 3;
                                                                                                                              scalingMethod,

                                                                                                                              let inputShapesHtml = "";
                                                                                                                              tr.appendChild(typeCell);

                                                                                                                              return `Actual: ${context.parsed.x.toFixed(4)}, Predicted: ${context.parsed.y.toFixed(4)}`;

                                                                                                                              layers[currentLayer] = [];


                                                                                                                              "rgba(255, 99, 132, 0.7)",
                                                                                                                              details = `Kernel: ${layer.kernel_size.join("Ã—")}`;
                                                                                                                          }
                                                                                                                              <span class="info-value">${summary.node_count || summary.nodes?.length || 0}</span>
                                                                                                  },
                                                                                                                  <div class={"info-card">
                                                                                                                    if (nextButton) nextButton.disabled = currentPage >= totalPages - 1;
                                                                                                                  let pageSize = 10;
                                                                                                                      try {
                                                                                                                          let currentPage = 0;
                                                                                                                                console.error("Model testing error:", result.error);
                                                                                                                                      indexCell.className = "row-index";
                                                                                                                                        for (let offset = 0; offset < byteCharacters.length; offset += 512) {
                                                                                                                                              const labelsContainer = document.getElementById("three-labels-container");
                                                                                                                                                      </tr>`;

                                                                                                                                                            transparent: true,
                                                                                                                                                                    return;
                                                                                                                                                                          layersTableBody.innerHTML =
                                                                                                                                                                                  ],


                                                                                                                                                                                              label: function (context) {
                                                                                                                                                                                                        data: results.map((item) => ({ x: item.actual, y: item.predicted })),

                                                                                                                                                                                                          labelsContainer.style.pointerEvents = "none";
                                                                                                                                                                                                                  </div>
                                                                                                                                                                                                                        columns.forEach((column) => {
                                                                                                                                                                                                                              showSpinner(false);
                                                                                                                                                                                                                                    return;
                                                                                                                                                                                                                                                            </div>
                                                                                                                                                                                                                                                              labelsContainer.style.left = "0";

                                                                                                                                                                                                                                                                    indexCell.className = "row-index";
                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                            pane.classList.remove("active");

                                                                                                                                                                                                                                                                                      title: {
                                                                                                                                                                                                                                                                                                title: {
                                                                                                                                                                                                                                                                                                            </ul>


                                                                                                                                                                                                                                                                                                              }
                                                                                                                                                                                                                                                                                                                const missingCount = document.getElementById("data-missing-count");
                                                                                                                                                                                                                                                                                                                    updateStatus("Feature selectors not found");

                                                                                                                                                                                                                                                                                                                          const totalRow = document.createElement("tr");
                                                                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                                                                nextPageButton.disabled = currentPage >= totalPages - 1;
                                                                                                                                                                                                                                                                                                                                            <p><strong>Producer:</strong> ${summary.producer || "Unknown"}</p>


                                                                                                                                                                                                                                                                                                                                                } catch (error) {
                                                                                                                                                                                                                                                                                                                                                                </div>
                                                                                                                                                                                                                                                                                                                                                                      opacity: 0.9,

                                                                                                                                                                                                                                                                                                                                                                        labelsContainer.style.position = "absolute";
                                                                                                                                                                                                                                                                                                                                                                              `Creating 2D visualization with ${graphData.nodes.length} nodes`,
                                                                                                                                                                                                                                                                                                                                                                                          </div>

                                                                                                                                                                                                                                                                                                                                                                                          function switchToMode(mode) {
                                                                                                                                                                                                                                                                                                                                                                                                const base64 = arrayBufferToBase64(arrayBuffer);
                                                                                                                                                                                                                                                                                                                                                                                                      });
                                                                                                                                                                                                                                                                                                                                                                                                          if (csvInput) {
                                                                                                                                                                                                                                                                                                                                                                                                                updateStatus(`Model loaded successfully: ${file.name}`);
                                                                                                                                                                                                                                                                                                                                                                                                                      renderCsvTable(pageData.data);
                                                                                                                                                                                                                                                                                                                                                                                                                            csvData.rowCount = result.rows;
                                                                                                                                                                                                                                                                                                                                                                                                                                headerRow.appendChild(indexHeader);

                                                                                                                                                                                                                                                                                                                                                                                                                                      console.log("Final visibility check completed");
                                                                                                                                                                                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                                                                                                                                                                              if (pageInfo)
                                                                                                                                                                                                                                                                                                                                                                                                                                                    const checkbox = document.createElement("input");
                                                                                                                                                                                                                                                                                                                                                                                                                                                      }
                                                                                                                                                                                                                                                                                                                                                                                                                                                        async function handleModelUpload(event) {

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                <strong>${input.name}</strong>

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    if (!positions[node.id]) return;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          summary.outputs.forEach((output) => {

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              cancelAnimationFrame(animationFrameId);

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                async function getArchitectureSuggestions() {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  if (!inputFeatures || !targetFeature) {

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        network.redraw();

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          predictionChart = new Chart(ctx, {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            labelsContainer.style.left = "0";
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  byteArrays.push(byteArray);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      return;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          nodes.forEach((node) => {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            function updatePaginationInfo() {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  if (summary.outputs) {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              title: {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              let csvData = null;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  </div>

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        showSpinner(false);

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              totalLabelCell.textContent = "Total Operations";
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              }

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  const toPos = positions[edge.to];

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      if (network) {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "Input",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  label.appendChild(span);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    });


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            return;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              console.log("CSV columns:", columns);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    `Creating 2D visualization with ${graphData.nodes.length} nodes`,


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  display: true,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                },
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    showError(`Postprocessing failed: ${error.message}`);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          view2dButton.addEventListener("click", () => {


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  borderWidth: 1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              <strong>Producer:</strong> ${summary.producer || "Unknown"}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    prevButton.disabled = currentPage === 0;


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                display: true,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      console.error("2D network container not found");



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        const tableView = document.getElementById("data-table-view");
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            showSpinner(false);

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    },

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      function setupEventListeners() {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          testSizeSlider = document.getElementById("test-size");
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    label.appendChild(checkbox);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        document.getElementById("metric-mae").textContent = formatNumber(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            targetOption.value = column;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                updateDataPreview(result.data);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      console.log("Network stabilized");
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          labelDiv.className = "node-label";
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            } else {

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              }

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  `);
                                                                          const modelInfoContainer = document.getElementById("model-info");
                                                                              return btoa(binary);
                                                                                  const span = document.createElement("span");
                                                                                    console.log(`Found UI elements:
                                                                                                        <li class="model-io-item">
                                                                                                                <div class="detail-item">
                                                                                                                      data: {
                                                                                                                                rows: "-",

                                                                                                                                    columns.forEach((column) => {
                                                                                                                                        const material = new THREE.LineBasicMaterial({
                                                                                                                                              },
                                                                                                                                                      csvInput.click();
                                                                                                                                                            modelDetailsElement.innerHTML = `
                                                                                                                          if (modelDetailsElement) {
                                                                                                                      function setupThreeJsScene(container) {
                                                                                                                      points.push(new THREE.Vector3(fromPos.x, fromPos.y, fromPos.z));
                                                                                                                      levelSeparation: 150,
                                                                                                                      scene.add(backLight);
                                                                                                                  }

                                                                                                                      return;
                                                                                                                  }
                                                                                                                  }
                                                                                                                          row.appendChild(countCell);
                                                                                                                            return;
                                                                                                                                        nameCell.textContent = node.name || "Unnamed";
                                                                                                                                        function createLabels(nodes, positions) {


                                                                                                                      const container = document.getElementById("network-2d");
                                                                                                                  };
                                                                                                                                                            <div class="layer-info">
                                                                                                                  }
                                                                                                                      },
                                                                                                                      function createNetworkVisualization(graphData) {
                                                                                                                            detailsCell.textContent = details;
                                                                                                                  } catch (error) {
                                                                                                                                              <th>Producer</th>
                                                                                                                  function populateOptimizationFeatures() {
                                                                                                                                  <div class="info-row">
                                                                                                                        console.log("Populating optimization features");

                                                                                                                        if (rowCount) rowCount.textContent = summary.rows || "-";
                                                                                                                          checkbox.value = column;
                                                                                                                          );

                                                                                                                            view3dButton.addEventListener("click", () => {
                                                                                                                                          if (!file) return;
                                                                                                                  },
                                                                                                                                      </div>
                                                                                                          if (modelInput) {
                                                                                                          }
                                                                                                                  td.textContent = row[column] !== null ? row[column] : "";
                                                                                                                      </div>
                                                                                                  }
                                                                                                      return;
                                                                                                                                          <span class={"info-value">${summary.producer || "Unknown"}</span>
                                                                                                                                                  '<tr><td colspan="3">No layer information available</td></tr>';
                                                                                                                                                results.metrics.mae,
                                                                                                                                            function checkEnableTestButton() {
                                                                                                                                                            scalingMethod,
                                                                                                                                                              ${(summary.inputs || []).map((i) => `<li>${i.name} - ${i.type}</li>`).join("")}
                                                                                                                                                              </div>

                                                                                                                                                controls.update();
                                                                                                                                                  function readFileAsArrayBuffer(file) {
                                                                                                                                                          if (!file) return;
                                                                                                                                                                  html += `<tr>
                                                                                                                                                                        showSpinner(false);
                                                                                                                                                                        }

                                                                                                                                                                          function showSpinner(show) {
                                                                                                                                                                                              <div class="detail-item">
                                                                                                                                                                                                  setTimeout(() => {
                                                                                                                                                                                                      if (view3dButton) {
                                                                                                                                                                                                          const prevPageButton = document.getElementById("prev-page");
                                                                                                                                                                                                                    },
                                                                                                                                                                                                                        }
                                                                                                                                                                                                                                  callbacks: {
                                                                                                                                                                                                                                            title: {
                                                                                                                                                                                                                                                    testSizeValue.textContent = `${Math.round(value * 100)}%`;

                                                                                                                                                                                                                                                          headerRow.appendChild(th);
                                                                                                                                                                                                                                                            const nextPageButton = document.getElementById("next-page");
                                                                                                                                                                                                                                                            window.addEventListener("resize", function () {
                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                  showError(result.error);

                                                                                                                                                                                                                                                                  function animate() {
                                                                                                                                                                                                                                                                        inputsHtml += '<ul class="model-io-list">';
                                                                                                                                                                                                                                                                              modelDetailsElement.innerHTML = "No model details available";


                                                                                                                                                                                                                                                                                  const layersTableBody = document.querySelector("#layers-table tbody");
                                                                                                                                                                                                                                                                                          const opType = node.op_type || "Unknown";
                                                                                                                                                                                                                                                                                            const nodeTypes = {};
                                                                                                                                                                                                                                                                                                    opTypeCounts[opType] = (opTypeCounts[opType] || 0) + 1;
                                                                                                                                                                                                                                                                                                          const label = document.createElement("label");

                                                                                                                                                                                                                                                                                                            if (!container2d || !container3d) {
                                                                                                                                                                                                                                                                                                                const result = await window.eel.postprocess_data(processingSteps)();
                                                                                                                                                                                                                                                                                                                        html += `<li class="model-io-item">
                                                                                                                                                    : typeColors[node.type || "unknown"] || 0x4f46e5;
                                                                                                                                                                      html += "</div>";
                                                                                                                                                                          if (loadCsvButton && csvInput) {
                                                                                                                                                                                      margin: 10,
                                                                                                                                                    layersTableBody.innerHTML =

                                                                                                                                                    const offsetY = (i - (nodesInLayer.length - 1) / 2) * nodeSpacing;
                                                                                                                                                                                      async function runModelTest(testSizeSlider) {
                                                                                                                                                                                              a.download = result.filename;
                                                                                                                                                                                                        borderColor: "rgba(75, 192, 192, 1)",
                                                                                                                                                updateLabelsPositions();
                                                                                                                                                                                                          targetSelect.innerHTML = '<option value="">-- Select Target --</option>';
                                                                                                                                                                                                            async function generateModelSummary() {

                                                                                                                                                                                                                      nodeObjects = {};
                                                                                                                                                                                                                                      </div>
                                                                                                                                                function updateModelDetails(summary) {
                                                                                                                                          }
                                                                                                                                              if (modelInput) {
                                                                                                                                                          td.dataset.column = column;
                                                                                                                                                            function showSpinner(show) {


                                                                                                                                                                    const file = event.target.files[0];
                                                                                                                                                                          displayArchitectureSuggestions(result.architecture);
                                                                                                                                                                              pageData.forEach((row) => {
                                                                                                                                                                                    handleTabChange("tab-visualize");
                                                                                                                                                                                                    <div class="detail-section">

                                                                                                                                                  if (!positions[edge.from] || !positions[edge.to]) {
                                                                                                                                          }
                                                                                                                                              }
                                                                                                                                                    outputsHtml += "</ul>";" +
                                                                                                      "" +
                                                                                                      "  const labelsContainer = document.getElementById("three-labels-container");" +
                                                                                                      "  Object.values(nodeObjects).forEach((node) => {" +
                                                                                                      "      const zIndex = Math.floor(1000 - distance);" +
                                                                                                      "    inputsContainer.innerHTML = inputsHtml;" +
                                                                                                      "    const downloadModelButton = document.getElementById("download-model");" +
                                                                                                      "    inputOption.textContent = column;" +
                                                                                                      "    if (prevPageButton) {" +
                                                                                                      "" +
                                                                                                      "              }," +
                                                                                                      "    const featureCheckboxes = document.getElementById("feature-checkboxes");" +
                                                                                                      "    const clearDataButton = document.getElementById("btn-clear-data");" +
                                                                                                      "  if (columnCount) columnCount.textContent = summary.columns || "-";" +
                                                                                                      "      };" +
                                                                                                      "" +
                                                                                                      "      });" +
                                                                                                      "" +
                                                                                                      "          enabled: true," +
                                                                                                      "" +
                                                                                                      "              label: function (context) {" +
                                                                                                      "      return;" +
                                                                                                      "  try {" +
                                                                                                      "    }" +
                                                                                                      "});" +
                                                                                                      "}" +
                                                                                                      "      if (result.error) {" +
                                                                                                      "  }" +
                                                                                                      "    try {" +
                                                                                                      "    container3d.innerHTML = "";" +
                                                                                                      "    status.textContent = message;" +
                                                                                                      "      color: color," +
                                                                                                      "" +
                                                                                                      "    thead.appendChild(headerRow);" +
                                                                                                      "      const summary = await window.eel.generate_model_summary()();" +
                                                                                                      "    labelObjects[node.id] = labelDiv;" +
                                                                                                      "" +
                                                                                                      "    document.body.style.cursor = show ? "wait" : "default";" +
                                                                                                      "            layersTableBody.appendChild(row);" +
                                                                                                      "" +
                                                                                                      "    if (emptyState) emptyState.style.display = "flex";" +
                                                                                                      "  return new Blob(byteArrays, { type: mimeType });" +
                                                                                                      "  if (!container) return;" +
                                                                                                      "        </tr>`;" +
                                                                                                      "    updateStatus(`Error: ${error.message}`);" +
                                                                                                      "    }" +
                                                                                                      "      console.error("2D network container not found");" +
                                                                                                      "" +
                                                                                                      "    const byteCharacters = atob(result.base64);" +
                                                                                                      "      const byteNumbers = new Array(slice.length);" +
                                                                                                      "" +
                                                                                                      "    inverse_scaling: document.getElementById("inverse-scaling").checked," +
                                                                                                      "" +
                                                                                                      "" +
                                                                                                      ""
                                                                                                                                          }
                                                                                                                                          })
                                                                                                                                          }}
                                                                                                                  }
                                                                                                                      </th>
                                                                                                                  }
                                                                                                                  })
                                                                                                                  }}
                                                                                                                          }
                                                                                                                          }</h4>
                                                                                                                          }
                                                                                                                          }
                                                                                                      }
                                                                                                  }
                                                                                          })
                                                                                }
                                                                        })
                                                                }
                                                }
                                      })
                                      })
                                        }
                                })
                                      })
                            }</strong></td>
                                                                              }
                                                                        }))</strong></td>
                                                                                              })
                                                                                                                      }
                                                                                                            }
                                                                                        }
                                                                  }
                                                              }
                                    })}</
                                                  }h4>
                                                          }
                                                                                          ]</strong>
                                                                                              }
                                                              })`
                                                                }}
                                                                                                                    }
                                    
                                        try {
                                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                                         const complexity = document.querySelector(
                                                                                                                                                                                                                                                                                                                                                 }
                                                                                                                                                                                                                                                                                                                                                             <h3>Recommended Architecture</h3>
                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                             }
                                                                                                                                                                                                                                                                                                                                                               container.style.position = "relative";
                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                         )
                                                                                                                                                                                                                                                                                                                                                                         }
                                                                                                                                                                                                                                                                                                                                     }
                                                                      }
</div></p>`}
</div></strong>`
                                                                              })}</h4>
                                                                    }
                                                              }
                                                          }
                                              })
                                    })
                                          })</h4>
                                  })}
                                                      }
                                })</strong></td>
                                        }
                                                  )
                                    }
                            }))</strong></p></strong>
                                          }
                            })
                      })
                    }</h3>
        }
      ]</strong></p>
    })</h4>
                  }</strong></p>
                })</h4>}
                    }
                </ul>
                      }
    })</strong>
                      }
                  }
          }
</div></table>
</div></h4>`</strong>