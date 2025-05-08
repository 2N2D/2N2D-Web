  if (headerRow) {
      columns: "-",

  }

      const material = new THREE.LineBasicMaterial({}
                  const value = parseFloat(testSizeSlider.value);
          const summary = await window.eel.generate_model_summary()();
              console.error("No data available for features");
                      runModelTest(testSizeSlider);

                              opTypeCounts[opType] = (opTypeCounts[opType] || 0) + 1;
                                  const layerLabels = [

                                              tr.appendChild(td);
                                  function updatePaginationInfo(currentPage, totalPages) {
                                            testSize,

                                                      reader.readAsArrayBuffer(file);
                                                      if (currentPage < totalPages - 1) {
                                                      }, 500);
        return;
                                  }
                                    const inputFeatures = document.getElementById("opt-input-features");

                                              </div>

        label.style.opacity = opacity.toFixed(2);
          const columns = Object.keys(csvData.data[0]);
              if (summary.outputs && summary.outputs.length > 0) {
                  window.csvData.data.length === 0
                  let binary = "";
                  thead.innerHTML = "";

                  tbody.appendChild(tr);
              } finally {
                      if (runTestButton) runTestButton.disabled = false;
                          const layerX = (layerIndex - layerKeys.length / 2) * layerSpacing;
  )();
    const opTypeCounts = {};


  }

              let threeJsRenderer = null;
  }
              nextLayerNodes.push(edge.to);

    updateStatus("Model test completed successfully");
          summary.outputs.forEach((output) => {
              opacity: 0.9,
              const file = event.target.files[0];
          }

                          <div class={"info-row">
                                .filter((node) => !incomingEdges[node.id])
                              tabButtons.forEach((button) => {
                              "Output",
                              return;
                              let scene, camera, renderer, controls;
                              th.textContent = column;
                              <h4>Basic Info</div>h4>
                              setupThreeJsScene(container);

                              updateStatus(`Error: ${error.message}`);
                          }

                          },
          const tabId = button.id;
                                  tabButton.click();
                                      const clearDataButton = document.getElementById("btn-clear-data");
                                        targetSelect.innerHTML = '<option value="">-- Select Target --</option>';
                                                hierarchical: {
                                                            datasets: [
                                                                  document.querySelectorAll(".tab-pane").forEach((pane) => {


                                                                      loadCsvButton.addEventListener("click", () => {

                                                                          waitForEel();
                                                                      }
                                                                  catch
                                                                      (error)
                                                                      {
                                                                      }
                                                                  )
                                                                      ;

                                                                      const indexCell = document.createElement("td");
                                                                      displayModelInfo(summary.model_info);

                                                                      const view3dButton = document.getElementById("view-3d");
                                                                      row.appendChild(typeCell);

                                                                      scene.background = new THREE.Color(0xffffff);
                                                                      updateStatus(`Error: ${result.error}`);
                                                                      if (missingCount) missingCount.textContent = totalMissing;
                                                                  }

                                                                        if (result.error) {
                                                                                  columns.forEach((column) => {
                                                                                  }
                                                                                      if (!layersTableBody) return;
                                                                                        ];
                                                                const modelDetailsElement = document.getElementById("model-details");
                                                                    return;
                                                                        if (!layersTableBody) return;
                                                                              const typeCell = document.createElement("td");
                                                                                  try {
                                                                                      targetSelect.appendChild(option);
                                                                                      `Creating 2D visualization with ${graphData.nodes.length} nodes`,

                                                                                  }
                                                                                      const type = node.type || "unknown";


                                                                        )();
                                                                                      const slice = byteCharacters.slice(offset, offset + 512);

                                                                                          URL.revokeObjectURL(url);
                                                                                            try {
                                                                                                    testSizeSlider = document.getElementById("test-size");


                                                                                                        container3d.classList.remove("active");
                                                                                                          const ctx = document.getElementById("predictions-chart").getContext("2d");
                                                                                                                let details = "";
                                                                                                                    indexHeader.textContent = "#";
                                                                                                                            summary.inputs.forEach((input) => {



                                                                                                                                                    </tr>
                                                                                                                                    targetOption.textContent = column;
                                                                                                                            } catch (error) {
                                                                                                      "Output",
                                                                                                          let totalPages = 0;
                                                                                                                  text: "Actual Values",
                                                                                                                        function updatePaginationInfo() {
                                                                                                                        } finally {

                                                                                                                        animate();
                                                                                                                                z: 0,
                                                                                                                                        container2d.style.display = "block";
                                                                                                                                    showSpinner(false);
                                                                                                                                            updateDataSummary({

                                                                                                                                                      checkbox.addEventListener("change", checkEnableTestButton);



                                                                                                                                                                    text: "Number of Neurons",

                                                                                                                                                      showSpinner(false);
                                                                                                                                                                        showSpinner(true);

                                                                                                                                                                              if (result.error) {
                                                                                                                                                                                          td.dataset.column = column;
                                                                                                                                                                                              const modelInput = document.getElementById("model-input");
                                                                                                                                                                                                        currentPage--;

                                                                                                                                                                                                            modelDetailsElement.innerHTML = html;
                                                                                                                                                                                                                      borderWidth: 1,
                                                                                                                                                                                                                              if (!runTestButton) return;
                                                                                                                                                                                                                                      x: layerX,
                                                                                                                                                                                                                                                "Input",
                                                                                                                                                                                                                                            if (inputsContainer) {
                                                                                                                                                                                                                                                    const targetSelect = document.getElementById("target-select");

                                                                                                                                                                                                                                                                        <li class={"model-io-item">

                                                                                                                                                                                                                                                                            html += `<div class="detail-section">
    if (summary.inputs && summary.inputs.length > 0) {
    inverse_scaling: document.getElementById("inverse-scaling").checked,
    }
    const url = URL.createObjectURL(blob);


    }
    targetFeature.appendChild(targetOption);
    try {


    const clearDataButton = document.getElementById("btn-clear-data");

      modelDetailsElement.innerHTML = `
                                                                                                                                                                                                                                                                              window.graphData = result;
                                                                                                                                                                                                                                                                              html += "</tbody></table>";
                                                                                                                                                                                                                                                                                  console.log(`Moving to page ${currentPage + 1}`);
                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                  return;
                                                                                                                                                                                                                                                    const pageInfo = document.getElementById("page-info");

                                                                                                                                                                                                                                                                                      const nodesToProcess = nodeServerPages    const targetSelect = document.getElementById("target-select");
                                                                                                                                                                                                                                                                                            loadCsvButton.addEventListener("click", () => {
                                                                                                                                                                                                                                                                                                      const scalingMethod = document.getElementById("scaling-method").value;
                                                                                                                                                                                                                                                                                                        currentViewMode = mode;
                                                                                                                                                                                                                                                                                                                        <div class={"info-row">
                                                                                                                                                                                                                                                                                                                                            </div>
                                                                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                                                            });

                                                                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                                                                  scene = null;
                                                                                                                                                                                                                                                      checkbox.value = column.name;
                                                                                                                                                                                                                                                                  data: layers,
                                                                                                                                                                                                                                                                          a.href = url;


                                                                                                                                                                                                                                                                      console.log("No graph data available");
                                                                                                                                                                                                                                                                            ...architecture.hidden_layers,
                                                                                                                                                                                                                                            } finally {
                                                                                                                                                                                                                                                  targetSelect.disabled = false;
                                                                                                                                                                                                                                                      columnsInfo.forEach((column) => {

                                                                                                                                                                                                                                                                nextButton.disabled = currentPage >= totalPages - 1;
                                                                                                                                                                                                                                                                  if (!container) return;
                                                                                                                                                                                                                                                                    const runTestButton = document.getElementById("run-test");
                                                                                                                                                                                                                                                                        targetSelect.disabled = false;
                                                                                                                                                                                                                                                                          const view3dButton = document.getElementById("view-3d");
                                                                                                                                                                                                                                                                              updateOptimizationUI();
                                                                                                                                                                                                                                                                                console.log("Updating model testing UI...");

                                                                                                                                                                                                                                                                                                </div>
                                                                                                                                                                                                                                                      },
                                                                                                                                                                                                                                                                console.error("Error loading page:", error);
                                                                                                                                                                                                                                                const blob = new Blob(byteArrays, { type: "application/octet-stream" });
                                                                                                                                                                                                                                                        network.fit();
                                                                                                                                                                                                                                                                typeCell.textContent = opType;
                                                                                                                                                                                                                                                                  const nextPageButton = document.getElementById("next-page");

                                                                                                                                                                                                                                                                          updateStatus(`Error: ${pageData.error}`);


                                                                                                                                                                                  });

                                                                                                                                                                                  { x: min, y: min },
                                                                                                                                                                                    console.log(`Updating pagination: page ${currentPage + 1} of ${totalPages}`);
                                                                                                                                                                              }

                                                                                                                                                                                camera = new THREE.PerspectiveCamera(40, aspect, 1, 10000);

                                                                                                                        controls.rotateSpeed = 0.5;
                                                                                                                              updateStatus(`Error: ${result.error}`);
                                                                                                                                  const thead = table.querySelector("thead");
                                                                                                                                  let animationFrameId = null;
                                                                                                                                    const nodeTypes = {};
                                                                                                                                        if (nextPageButton) {
                                                                                                                                                      backgroundColor: "rgba(75, 192, 192, 0.6)",

                                                                                                                                                              const hiddenLayersStr = architecture.hidden_layers.join(", ");
                                                                                                                                                            data: {
                                                                                                                                                                <strong>Node
                                                                                                                                                                    Count:</strong>
                                                                                                                                                                $
                                                                                                                                                                {
                                                                                                                                                                    summary.node_count || summary.nodes?.length || 0
                                                                                                                                                                }
                                                                                                                                                                controls.zoomSpeed = 1.0;
                                                                                                                                                                nodesToProcess.push(...nextLayerNodes);
                                                                                                                                                                return;


                                                                                                                                                                modelDetailsElement.innerHTML = html;


                                                                                                                                                            });
                                                                                                                                        }

                                                                                                                          y: {
                                                                                                                          }

                                                                                                                          x: {
                                                                                                                          } finally {
                                                                                                                                          const view3dButton = document.getElementById("view-3d");
                                                                                                                                            const opTypeCounts = {};
                                                                                                                                                  rows: "-",
                                                                                                            const line = new THREE.Line(geometry, material);
                                                                                                                                                y: {
                                                                                                                                                          const totalPages = Math.ceil(window.csvData.data.length / pageSize);
                                                                                                                                                                        <th>Version</th>
                                                                                                                                                  const positions = {};
                                                                                                                                                      setTimeout(() => {


                                                                                                                                                      });


                                                                                                                                                        if (tabButton) {
                                                                                                                                                            updateStatus(`Error: ${pageData.error}`);
                                                                                                                                                            updateOptimizationUI();
                                                                                                                                                        }
                                                                                                                                                );
                                                                                                                                                const url = URL.createObjectURL(blob);
                                                                                                                                                }
                                                                                                                                                    if (result.error) {
                                                                                                                                                    ),
                                                                                                                                                    }
                                                                                                                                                                <h3>Recommended Architecture</h3>
                                                                                                                                                console.error("Error generating summary:", summary.error);
                                                                                                                                              inputsHtml += "</ul>";
                                                                                                                                              console.warn(`Missing positions for edge ${edge.from} -> ${edge.to}`);

                                                                                                                                            if (testingContent) testingContent.style.display = "block";
                                                                                                                                                  pointRadius: 0,
                                                                                                            labelsContainer.remove();
                                                                                                                                            nextPageButton.disabled = currentPage >= totalPages - 1;
                                                                                                                                              console.log("Model input event handler added");
                                                                                                                                              tr.appendChild(detailsCell);

                                                                                                                                                details = `Shape: [${layer.shape.join(", ")}]`;
                                                                                                                                          const rowCount = document.getElementById("data-row-count");


                                                                                                        }
                                                                                                                    const canvasElements = container.querySelectorAll("canvas");
                                                                                                                      thead.appendChild(headerRow);
                                                                                                                        binary += String.fromCharCode(uint8Array[i]);

                                                                                                    }
                                                                                                      ],

                                                                                                          maintainAspectRatio: false,
                                                                                                    }

                                                                          createEdges(data.edges, nodePositions);

                                                                          if (outputsContainer) {

                                                                                      const row = document.createElement("tr");
                                                                            if (renderer) {
                                                                                updateStatus("Generating architecture suggestions...");
                                                                            }
                                                                              pageData.forEach((row) => {
                                                                                        container.innerHTML =
                                                                                              if (tableBody) {
                                                                                                      document.getElementById("metric-r2").textContent = formatNumber(
                                                                                                                              </li>
                                                                                                        totalPages = Math.ceil(result.rows / pageSize);
                                                                                                            displayTestResults(result);
                                                                                                                const layerX = (layerIndex - layerKeys.length / 2) * layerSpacing;

                                                                                              }
                                                                                                    architectureChart.destroy();
                                                                                        springLength: 120,
                                                                                              const typeColors = {};
                                                                                            nodesInLayer.forEach((node, i) => {


                                                                                                    if (emptyState) emptyState.style.display = "flex";
                                                                                                        container3d.innerHTML = "";
                                                                                                          renderer.setSize(width, height);

                                                                                                                    borderColor: "rgba(255, 99, 132, 0.7)",
                                                                                                                            const preprocessButton = document.getElementById("btn-preprocess");
                                                                                                                      ];


        </div>

                                                                                                  controls.dampingFactor = 0.12;

                                                                                                                  </div>
                                                                                            });
                                                                                                  for (let i = 0; i < slice.length; i++) {
                                                                                                          labelsContainer.remove();
                                                                                                                    label: "Predictions",
                                                                                                                                    </tr>`;
                                                                                                                                        if (nextPageButton) {
                                                                                                                                                  tooltip: {
                                                                                                                                                  
                                                                                                                                                      try {
                                                                                                                                                          tabButtons.forEach((button) => {
                                                                                                                                                          
                                                                                                                                                                      `;
                                                                                                                                      if (renderer) {
                                                                                                                                                updateModelDetails(result.summary);
                                                                                                                                                  const emptyState = document.getElementById("data-empty-state");
                                                                                                                                                          console.log("Clear data button enabled");
                                                                                                                                                                              <div class={"layer-info">
                                                                                                                                                                                const contentId = "content-" + tabId.replace("tab-", "");


                                                                                                                                                                                  const edges = new vis.DataSet(graphData.edges);
                                                                                                                                                                                  return;
                                                                                                                                                                                    totalRow.appendChild(totalCountCell);
                                                                                                                                                                                    positions[node.id] = {
                                                                                                                                                                                          detailsElement.innerHTML = html;

                                                                                                                                                                                          function displayModelSummary(summary) {
                                                                                                                                                                                                    displayArchitectureSuggestions(result.architecture);
                                                                                                                                                                                                                const row = document.createElement("tr");
                                                                                                                                                                                                                  layerKeys.forEach((layerIndex) => {

                                                                                                                                                                                                                          data.forEach((row, index) => {

                                                                                                                                                                                  reader.readAsArrayBuffer(file);
                                                                                                                                                                              }
                                                                                                                                                                                    summary.inputs.forEach((input) => {
                                                                                                                                                                                            if (summary.outputs && summary.outputs.length > 0) {
                                                                                                                                                                              },

                                                                                                                                                                                  let csvData = null;
                                                                                                                                                                                                  window.optimizedModelPath,
                                                                                                                                                                                      if (!positions[node.id]) {
                                                                                                                                                                              }
                                                                                                                                                                                          columnsInfo.forEach((column) => {
                                                                                                                                                                              }
                                                                                                                                                                                renderer.domElement.style.zIndex = "1";
                                                                                                                                                                                              updateStatus("Model test completed successfully");
                                                                                                                                                                                                    for (let i = 0; i < slice.length; i++) {
                                                                                                                                                                                                                summary.inputs.forEach((input) => {
                                                                                                                                                                                                                        container3d.style.display = "none";
                                                                                                                                                                                                                            console.error("Missing containers");
                                                                                                                                                                                                                                const result = await window.eel.test_model(

                                                                                                                                                                              }
                                                                                                                                                                                  targetFeature.appendChild(targetOption);
                                                                                                                                                                                                                    const text = node.label || node.id.toString();

                                                                                                                                                                                                                                            <td>${modelInfo.producer || "Unknown"}</td>

                                                                                                                                      } finally {

                                                                                                                                                                                                                outputsHtml += `
                                                                                                                                                                                                                
                                                                                                                                                                                                                    }
                                                                                                                                                                                                                    
                                                                                                                                                                                                                      }
                                                                                                                                                                                                                              summary.inputs.forEach((input) => {
                                                                                                                                                                                                                              
                                                                                                                                                                                                                              
                                                                                                                                                                                                                                  }
                                                                                                                                                                                                                                      const modelDetailsElement = document.getElementById("model-details");
                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                            columns.forEach((column) => {
                                                                                                                                                                                                                                                container3d.style.display = "block";
                                                                                                                                                                                                                                                  nodes.forEach((node) => {
                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                  }
                                                                                                                                                                                                                                                        updateStatus("Architecture suggestions generated");
                                                                                                                                                                                                                                                          function readFileAsArrayBuffer(file) {
                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                  if (tableView) tableView.style.display = "none";
                                                                                                                                                                                                                                                                        scales: {
                                                                                                                                                                                                                                                                            if (nextButton) {
                                                                                                                                                                                                                                                                                return;
                                                                                                                                                                                                                                                                                        td.textContent = row[column] !== null ? row[column] : "";
                                                                                                                                                                                                                                                                                                                <strong>Activation Function:</strong> ${architecture.activation}


    console.error("Missing containers");
    }
      const zIndex = Math.floor(1000 - distance);
    layersTableBody.innerHTML = "";
    const layerLabels = [
                    <tr>
                    
                    
                                  ),
                                            if (currentPage < totalPages - 1) {
                                                  scene = null;
                                                    function createArchitectureVisualization(architecture) {
                                                        }
                                                            const type = node.type || "unknown";
                                                                const label = labelObjects[nodeId];
                                                                            const row = document.createElement("tr");
                                                                                  const totalCountCell = document.createElement("td");
                                                                                        Object.entries(opTypeCounts).forEach(([opType, count], index) => {
                                                                                              nodesToProcess.forEach((nodeId) => {
                                                                                                  );
                                                                                                        opacity: 0.9,
                                                                                                        
                                                                                                            labelsContainer.appendChild(labelDiv);
                                                                                                                    csvInput.click();
                                                                                                                        if (result.error) {
                                                                                                                              layersTableBody.innerHTML =
                                                                                                                                const pageInfo = document.getElementById("page-info");
                                                                                                                                  const targetSelect = document.getElementById("target-select");
                                                                                                                                      tbody.innerHTML = "";
                                                                                                                                          const material = new THREE.MeshLambertMaterial({
                                                                                                                                                totalRow.appendChild(totalLabelCell);
                                                                                                                                                          label: "Predictions",
                                                                                                                                                                  '<tr><td colspan="3">No layer information available</td></tr>';
                                                                                                                                                                  
                                                                                                                                                                    }
                                                                                                                                                                        });
                                                                                                                                                                            } finally {
                                                                                                                                                                                    console.log("Preprocess button enabled");
                                                                                                                                                                                      container3d.appendChild(labelsContainer);
                                                                                                                                                                                      
                                                                                                                                                                                          }
                                                                                                                                                                                              }
                                                                                                                                                                                                    showSpinner(true);
                                                                                                                                                                                                                            <td>${modelInfo.producer || "Unknown"}</td>
                                                                                                                                                                                                                                } else {
                                                                                                                                                                                                                                      tr.appendChild(detailsCell);
                                                                                                                                                                                                                                          }
                                                                                                                                                                                                                                            }
                                                                                                                                                                                                                                                        <h4>Inputs (${summary.inputs?.length || 0})</h4>
                                                                                                                                                                                                                                                            container3d.classList.remove("active");
                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                }
                                                                                                                                                                                                                                                                      updateStatus(`Model loaded successfully: ${file.name}`);
                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                          return btoa(binary);
                                                                                                                                                                                                                                                                                          ${shape ? `<span class="io-shape">${shape}</span>` : ""}
                                                                                                                                                                                                                                                                                                setTimeout(waitForEel, 100);
                                                                                                                                                                                                                                                                                                  if (tableBody) tableBody.innerHTML = "";
                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                        const slice = byteCharacters.slice(offset, offset + 512);
                                                                                                                                                                                                                                                                                                                    layersTableBody.appendChild(row);
                                                                                                                                                                                                                                                                                                                          label.style.zIndex = zIndex;
                                                                                                                                                                                                                                                                                                                              const loadCsvEmptyButton = document.getElementById("btn-load-csv-empty");
                                                                                                                                                                                                                                                                                                                              });
                                                                                                                                                                                                                                                                                                                                    controls = null;
                                                                                                                                                                                                                                                                                                                                    }
                                                                                                                                                                                                                                                                                                                                      labelsContainer.style.position = "absolute";
                                                                                                                                                                                                                                                                                                                                          const blob = new Blob(byteArrays, { type: "application/octet-stream" });
                                                                                                                                                                                                                                                                                                                                                    result.summary.nodes.forEach((node, index) => {
                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                        </tr>
}
    const ctx = canvas.getContext("2d");
      console.error("Error cleaning up THREE.js:", e);

    html += '<div class="detail-section"><h4>Outputs</h4>';
    function create3DNetworkVisualization(data) {
    function displayModelSummary(summary) {
      renderer.domElement.style.zIndex = "1";
                const totalPages = Math.ceil(window.csvData.data.length / pageSize);
                    });
                        layersTableBody.innerHTML = "";
                        
                            const modelInput = document.getElementById("model-input");
                                  }
                                  
                                  
                                    if (!csvData || !csvData.data || csvData.data.length === 0) {
                                          label.className = "feature-checkbox";
                                                    nodeDistance: 150,
                                                          updateStatus(
                                                                const offsetY = (i - (nodesInLayer.length - 1) / 2) * nodeSpacing;
                                                                        switchToMode("3d");
                                                                            const loadCsvButton = document.getElementById("btn-load-csv");
                                                                                  aspectRatio: 1.5,
                                                                                      return;
                                                                                      
                                                                                            updateStatus(`Error: ${error.message}`);
                                                                                                    smooth: { type: "cubicBezier", roundness: 0.5 },
                                                                                                    
                                                                                                      const fov = camera.fov * (Math.PI / 180);
                                                                                                              row.appendChild(indexCell);
                                                                                                              
                                                                                                                                      <td>${modelInfo.domain || "Unknown"}</td>

    thead.appendChild(headerRow);
  renderer.domElement.style.top = "40px";
  });


      const span = document.createElement("span");

  const inputFeaturesSelect = document.getElementById("input-features");
      html += '<ul class="model-io-list">';
                          `;
                                                                                                                                                                                                        const hasFeatures = featureCheckboxes.length > 0;
                                                                                                                                                                                                              y: {
                                                                                                                                                                                                                      }
                                                                                                                                                                                                                          !window.csvData.data ||
                                                                                                                                                                                                                                    pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
                                                                                                                                                                                                                  const headerRow = document.createElement("tr");
                                                                                                                                                                                                                    const targetFeature = targetFeatureSelect.value;
                                                                                                                                                                                                                        modelInfoContainer.innerHTML = `
                                                                                                                                                                                                                              totalPages = Math.ceil(result.rows / pageSize);
                                                                                                                                                                                                                                            ),
                                                                                                                                                                                                                                              }
                                                                                                                                                                                                                                                    outputsHtml += "<p>No output information available</p>";
  try {
    console.log(`Hidden tab: ${pane.id}`);
              springConstant: 0.01,
                  document.getElementById("metric-r2").textContent = formatNumber(
                  
                      }
                            const reader = new FileReader();
                                  });
                                  
                                    updateStatus("Running model test...");
                                              levelSeparation: 150,
                                                  html += "</div>";
    const assignedNodes = new Set();

          data: results.map((item) => ({ x: item.actual, y: item.predicted })),


    updateStatus(`Error: ${error.message}`);
    let labelObjects = {};
                <strong>IR Version:</strong> ${summary.ir_version || "Unknown"}
                    updateStatus("No optimized model available");
      renderer = new THREE.WebGLRenderer({
              if (emptyState) emptyState.style.display = "flex";
                nodes.forEach((node) => {
                        nodesInLayer.forEach((node, i) => {
                                  displayModelInfo(summary.model_info);
                                      if (prevButton) {
                                                return;
                                                        },
                                                              row.appendChild(cell);
                                      }
                                              },
                                                const columns = Object.keys(csvData.data[0]);
                    points.push(new THREE.Vector3(toPos.x, toPos.y, toPos.z));
                                  "rgba(54, 162, 235, 1)",
                                                      <div class={"param-info">
                                                              return;
                                                      
                                                      
                                                          typeColors[type] = colors[i % colors.length];
                                                                          </div>
                                                                              a.download = result.filename;
                                                                              
                                                                              
                                                                              
                                                                                    featureCheckboxes.appendChild(label);
                                                      });
                      </div>
    return;
                    </div>

                    </div>
        hierarchicalRepulsion: {
            
                        </ul>
                            }
                                nodesInLayer.forEach((node, i) => {
                                            updateStatus(`Error:resulterror}`);      forceCheckOptimizationTab();      updateModelDetails(resultsummary);    container3dinnerHTML = ;  if (pageInfo)    }    const fromPos = positions[edgefrom];        font:size:14 face:Inter }    consoleerror(`Tab content with ID contentId} not found`);  nodesforEach((node) =>   predictionChart = newChart(ctx                     div class=param-info>    showSpinner(false);              rgba(54 162 235 07)let scene camera renderer controls;            text:Predicted Values    if (!positions[nodeid]) return;    container3dstylewidth = 100%;          tooltip:Empty state:!!emptyState}  updateModelTestingUI();      consolewarn(`Missing positions for edge edgefrom} -> edgeto}`);        x:layerX                            span class=detail-type>outputdata_type}span>        consolelog(Previous page button clicked);        const layersTableBody = documentquerySelector(layers-table tbody);  const maxDim = Mathmax(sizex sizey sizez);    const pageData = csvDatadataslice(startIndex endIndex);    } else   const container = documentgetElementById(network-3d);    checkboxname = features;        testSizeValuetextContent = `Mathround(value * 100)}%`;      consolewarn(`Missing positions for edge edgefrom} -> edgeto}`);    const layers = [    targetSelectaddEventListener(change checkEnableTestButton);  const testingContent = documentgetElementById(testing-content);      documentgetElementById(content-optimize)style =      consolelog(Start optimization button event listener added);documentgetElementById(debugBtn)addEventListener(click function ()       labelappendChild(checkbox);    if (!data || datalength === 0)     if (loadCsvEmptyButton && csvInput)     theadinnerHTML = ;      });  const fov = camerafov * (MathPI  180);  return positions;                    inputShapesHtml || p>No input information availablep>}                        strong>Version:strong> modelInfoproducer_version || Unknown}                        inputtype ? `span class=io-type>inputtype}span>` :}        details = `Kernel:layerkernel_sizejoin(Ã—)}`;    const label = labelObjects[nodeId];  const directionalLight = newTHREEDirectionalLight(0xffffff 08);    html += `div class=detail-section>    const nodeMesh = nodeObjects[nodeId];    edgesforEach((edge) =>   if (inputFeatureslength === 0)         display:block !important;;    consolelog(No data to display);    consoleerror(Missing containers);    theadinnerHTML = ;      updateDataSummary(resultsummary);          pointHoverRadius:7        x:let architectureChart = null;      if (preprocessButton)     }    const view2dButton = documentgetElementById(view-2d);      trappendChild(indexCell);            div class=architecture-card>        updateStatus(`Error:resulterror}`);      const reader = newFileReader();    if (spinner)     }    layers[nodelayer]push(node);  function renderCsvTable(data)       const indexCell = documentcreateElement(td);    if (!layersTableBody) return;        html += `li class=model-io-item>                    li>    updateOptimizationUI();      displayArchitectureSuggestions(resultarchitecture);      updateOptimizationUI();      return;    const headerRow = documentcreateElement(tr);  if (tableBody) tableBodyinnerHTML = ;        div>      });    const hasModelAndData = network !== null && csvData !== null;      updateStatus(`Error:resulterror}`);      }                div class=architecture-details>      }                        strong>Batch Size:strong> architecturebatch_size}    });}    predictionChartdestroy();    updateStatus(Load a model first);    if (!positions[nodeid])       updateModelTestingUI();        consolelog(Switching to 2D mode);      switchToMode(currentViewMode);    if (summaryinputs && summaryinputslength > 0)       trappendChild(indexCell);    let currentLayer = 0;  labelsContainerstyletop = 0;      summarynodesforEach((node) =>   const view2dButton = documentgetElementById(view-2d);    container3dstyledisplay = none;                        strong>Hidden Layers:strong> architecturehidden_layerslength} layers [hiddenLayersStr}]      });      optiontextContent = columnname;  camera = newTHREEPerspectiveCamera(40 aspect 1 10000);      }  }      ]  const labelsContainer = documentgetElementById(three-labels-container);        updateStatus(`Error:resulterror}`);                strong>inputname || Unnamed}strong>    if (summarynodes && summarynodeslength > 0)       html += thead>tr>th>Operationth>th>Countth>tr>thead>;    container3dinnerHTML = ;  containerstyledisplay = block;    if (windowgraphData)   }                p>strong>Batch Size:strong> resultbatch_size}p>  if (!summary || !detailsElement) return;  const tableView = documentgetElementById(data-table-view);    pointspush(newTHREEVector3(fromPosx fromPosy fromPosz));      if (preprocessButton)         testSizeValuetextContent = `Mathround(value * 100)}%`;    const a = documentcreateElement(a);                p>strong>Learning Rate:strong> resultlearning_rate}p>    targetFeatureappendChild(targetOption);    if (clearDataButton)   rendererdomElementstyleposition = absolute;          rows:-  currentViewMode = mode;                    div>        outputsHtml += `      });    if (view3dButton)           fill:false            if (!testSizeSlider)     resultsContainerstyledisplay = block;        const shape = outputshape ? `[outputshapejoin( )}]` :;  const runTestButton = documentgetElementById(run-test);  return newBlob(byteArrays  type:mimeType });    columnsInfoforEach((column) =>       const totalRow = documentcreateElement(tr);  const min = Mathmin(allValues);  const hasModel = windowgraphData && windowgraphDatasummary;                    outputShapesHtml || p>No output information availablep>}    if (!modelDetailsElement) return;    const table = documentgetElementById(csv-table);    } catch (error)   if (activeButton)                     div>                p>strong>Batch Size:strong> resultbatch_size}p>          missing_values:}  consolelog(`Updating pagination:page currentPage + 1} of totalPages}`);  const nodePositions = calculateNodePositions(datanodes dataedges);  if (emptyState) emptyStatestyledisplay = none;                            strong>inputname}:strong> [inputshapejoin( )}]  windowgraphData = data;                    h4>Outputsh4>      architecturehidden_layers  }      });  if (nextPageButton)       const result = await windoweelload_csv_data(base64 filename)();  });  function renderCsvTable(data)   consolelog(`Container dimensions:width}xheight}`);        Testing content:!!testingContent}                div class=detail-section>      datasets:[      updateModelTestingUI();    labelclassName = feature-label;            p>strong>IR Version:strong> summaryir_version || Unknown}p>    container3dstylewidth = 100%;  if (tabButton)       showSpinner(true);      indexCellclassName = row-index;  function readFileAsArrayBuffer(file)     }    try       let inputShapesHtml = ;    return;    nextPageButtondisabled = currentPage >= totalPages - 1;  camerapositionset(0 0 2000);      if (summaryinputs)         scales:const layersTableBody = documentquerySelector(layers-table tbody);    updateStatus(`Error:errormessage}`);      if (clearDataButton)     }function setupThreeJsScene(container)       thtextContent = column;      nodeObjects = };function populateFeatureSelectors(columns)         return;  if (network)     showSuccess(resultmessage);      showSpinner(true);function animate()         div class=detail-item>  if (!testSizeSlider)       layersTableBodyappendChild(totalRow);            beginAtZero:true    tbodyinnerHTML = ;                    span class=info-value>summarynode_count || summarynodes?length || 0}span>          levelSeparation:150    !windowcsvDatadata ||    worldPosproject(camera);        const typeCell = documentcreateElement(td);  const pageInfo = documentgetElementById(page-info);                    tr>  }    indexHeadertextContent = ;                    div class=detail-item>            p>strong>IR Version:strong> summaryir_version || Unknown}p>function displayTestResults(results)     container3dstyledisplay = block;    } finally     if (!layers || layerslength === 0)   const emptyState = documentgetElementById(testing-empty-state);  const labelsContainer = documentgetElementById(three-labels-container);  controls = newTHREEOrbitControls(camera rendererdomElement);    setTimeout(() =>             ul>            tr>`;    modelDetailsElementinnerHTML = html;  if (!csvData || !csvDatadata || csvDatadatalength === 0)                         th>Domainth>      }    const fromPos = positions[edgefrom];}  }      consoleerror(Error generating model summary:error);            strong>IR Version:strong> summaryir_version || Unknown}      const tr = documentcreateElement(tr);    if (testingContent) testingContentstyledisplay = none;      summaryoutputsforEach((output) =>       csvInputaddEventListener(change handleCsvUpload);      });      });      createArchitectureVisualization(resultarchitecture);            ul>    consolelog(Event listeners set up successfully);      opacity:09      showSpinner(false);      }}      checkboxname = features;    return;    consolelog(`Hidden tab:paneid}`);        tddatasetrow = index;      updateStatus(`Error:errormessage}`);    cancelAnimationFrame(animationFrameId);          inputShapesHtml += `  const detailsElement = documentgetElementById(model-details);    const modelInput = documentgetElementById(model-input);    const tabButtons = documentquerySelectorAll(tab-button);      binary += StringfromCharCode(uint8Array[i]);  if (prevPageButton)   }  });  containerstyleposition = relative;    sphereuserData =  id:nodeid label:nodelabel type:nodetype };      html += ul class=model-io-list>;        const countCell = documentcreateElement(td);  let labelsContainer = documentgetElementById(three-labels-container);        updateDataSummary(function createNetworkVisualization(graphData)   }  controlsupdate();      });  const columns = Objectkeys(windowcsvDatadata[0]);    const layers = [        nodelayer = currentLayer;let scene camera renderer controls;});      html += `tr class=total-row>    URLrevokeObjectURL(url);    }      totalRowappendChild(totalLabelCell);          }                `;      updateStatus(Generating architecture suggestions);    });    featureContainerappendChild(label);    if (!table) return;      }}                div>    typeColors[type] = colors[i % colorslength];                    `;    if (loadCsvButton && csvInput)       readeronload = (e) => resolve(etargetresult);          }    consoleerror(Error running model test:error);  currentViewMode = mode;                div>        }        missingValues        });            indexCelltextContent = index + 1;  });  container3dappendChild(labelsContainer);  camerapositionset(0 0 2000);          centralGravity:00    targetSelectaddEventListener(change checkEnableTestButton);    const size = nodesize || 15;      const base64 = arrayBufferToBase64(arrayBuffer);                    span class=info-value>summaryproducer || Unknown}span>                        th>Domainth>              Array(architecturehidden_layerslength)fill(    documentgetElementById(metric-samples)textContent =    for (let i = 0; i  uint8ArraybyteLength; i++)     if (emptyState) emptyStatestyledisplay = flex;      architecturehidden_layersmap((_ i) => `Hidden i + 1}`)    } catch (error)             }        }        trappendChild(td);  renderersetSize(width height);      document        ]             x:min y:min }        updateStatus(`Error:resulterror}`);        }                    div class=param-info>        scalingMethod    const layerX = (layerIndex - layerKeyslength  2) * layerSpacing;                `;    const label = labelObjects[nodeId];    prevPageButtondisabled = currentPage === 0;  try           columns:-              label:function (context)       });  if (tabButton)       updateStatus(`Error:errormessage}`);  }      consoleerror(Error loading page:error);    const points = [];        csvInputclick();}    theadinnerHTML = ;      buttonaddEventListener(click () =>       layersTableBodyappendChild(tr);        const opType = nodeop_type || Unknown;    const slice = byteCharactersslice(offset offset + 512);    !windowcsvDatadata ||  scene = newTHREEScene();}  }          currentPage--;              return `Actual:contextparsedxtoFixed(4)} Predicted:contextparsedytoFixed(4)}`;    const thead = tablequerySelector(thead);      celltextContent = No layer information available;    } else   if (!csvData || !csvDatadata || csvDatadatalength === 0)         html += `tr>  consolelog(Displaying model summary:summary);function base64ToBlob(base64 mimeType)                 div>      headerRowappendChild(th);    } else   }  const maxDim = Mathmax(sizex sizey sizez);        display:block !important;;      const th = documentcreateElement(th);  waitForEel();          title:const loadCsvEmptyButton = documentgetElementById(btn-load-csv-empty);                        strong>Output Layer:strong> architectureoutput_layer} neurons  rendererrender(scene camera);      html += ul>;      showSpinner(true);    const tbody = tablequerySelector(tbody);  }  }      architecturehidden_layers            rowappendChild(indexCell);    if (windowgraphData)   if (!inputFeaturesSelect || !targetFeatureSelect)       updateStatus(Generating architecture suggestions);        ]      displayLayersInfo(summarylayers);      }        div>  controlsminDistance = 100;      headerRowappendChild(th);  });    const pageData = csvDatadataslice(startIndex endIndex);    );  function displayModelInfo(modelInfo)       let inputShapesHtml = ;  function createArchitectureVisualization(architecture)     const tabButtons = documentquerySelectorAll(tab-button);      const base64 = arrayBufferToBase64(arrayBuffer);  controlszoomSpeed = 10;            strong>Producer:strong> summaryproducer || Unknown}                        td>modelInfodomain || Unknown}td>      camera = null;      if (resulterror)       consolelog(Model input event handler added);                div>  sceneadd(directionalLight);  if (view3dButton) view3dButtonclassListtoggle(active mode === 3d);}    if (prevButton)       thtextContent = column;          columns:-  const layerSpacing = 200;      }    indexHeadertextContent = ;  }  animate();            display:true                }    inputFeaturesremove(0);          pointRadius:0        const shape = inputshape ? `[inputshapejoin( )}]` :;                    tr>      html += `tr class=total-row>  consolelog(`Container dimensions:width}xheight}`);                div class=detail-section>      positions[nodeid] =         y:offsetY  const nextPageButton = documentgetElementById(next-page);    }            currentPage++;      const offsetY = (i - (nodesInLayerlength - 1)  2) * nodeSpacing;      if (clearDataButton)           y:inputShapesHtml || p>No input information availablep>}      indexCelltextContent = currentPage * pageSize + index + 1;  const container3d = documentgetElementById(network-3d);      showSpinner(true);    }  const center = newTHREEVector3();      consolewarn(`No position found for node nodeid}`);    if (downloadModelButton)   containerstyledisplay = block;      nodeObjects = };                        strong>Batch Size:strong> architecturebatch_size}    const result = await windoweeldownload_optimized_model(        Testing content:!!testingContent}      controls = null;                table class=info-table>    if (summaryinputs && summaryinputslength > 0)     documentbodyremoveChild(a);  controls = newTHREEOrbitControls(camera rendererdomElement);  }        rowappendChild(typeCell);      const tr = documentcreateElement(tr);      detailsCelltextContent = details;            const typeCell = documentcreateElement(td);        div class=detail-section>    if (runTestButton) runTestButtondisabled = true;        inputsHtml += `    const hasModelAndData = network !== null && csvData !== null;  if (prevPageButton)       layersTableBodyappendChild(totalRow);      consoleerror(Error loading model:error);            const row = documentcreateElement(tr);    const pageData = csvDatadataslice(startIndex endIndex);    consolelog(No graph data available);      layersTableBodyappendChild(row);    predictionChartdestroy();          resultsummarynodesforEach((node index) =>   if (mode === 2d)   const testSize = parseFloat(testSizeSlidervalue);        if (emptyState) emptyStatestyledisplay = flex;    const byteCharacters = atob(resultbase64);  const targetFeature = documentgetElementById(opt-target-feature);  if (emptyState) emptyStatestyledisplay = none;  consolelog(`Found UI elements:const blob = newBlob(byteArrays  type:applicationoctet-stream });    if (!positions[nodeid])   const layerKeys = Objectkeys(layers)      const opTypeCounts = };        outputsHtml += `      };  directionalLightpositionset(1 1 1)normalize();    });    }        updateStatus(`Error:resulterror}`);    const featureCheckboxes = documentquerySelectorAll(      let inputShapesHtml = ;  const targetSelect = documentgetElementById(target-select);            div class=architecture-card>  } catch (error)     }    tabButtonsforEach((button) =>   async function getArchitectureSuggestions()       if (summaryerror)       });  camerapositionset(centerx centery centerz + distance);            beginAtZero:true    }        rowappendChild(countCell);  async function getArchitectureSuggestions()       prevPageButtonaddEventListener(click () => let nodeObjects = };                    div>  if (hasModel && hasData)       thtextContent = column;    updateStatus(Feature selectors not found);  if (layersTableBody)     const fromPos = positions[edgefrom];    resultsContainerstyledisplay = block;    const nextPageButton = documentgetElementById(next-page);      layersTableBodyappendChild(totalRow);  function readFileAsArrayBuffer(file)       nodes:text:Actual Values    try     consoleerror(`Tab content with ID contentId} not found`);      tbodyinnerHTML = tr>td colspan=100>No data availabletd>tr>;    labelstyletransform = translate(-50% -50%);                        strong>Input Layer:strong> architectureinput_layer} neurons          }  if (!inputFeatures || !targetFeature)       }    const modelInput = documentgetElementById(model-input);    const geometry = newTHREESphereGeometry(size 16 16);    const clearDataButton = documentgetElementById(btn-clear-data);      tbodyappendChild(tr);    if (modelInfoElement)     const prevPageButton = documentgetElementById(prev-page);                            strong>outputname}:strong> [outputshapejoin( )}]  labelsContainerstyleheight = 100%;          }        hierarchical:const canvasElements = containerquerySelectorAll(canvas);      displayArchitectureSuggestions(resultarchitecture);    nodesInLayerforEach((node i) =>     });    const span = documentcreateElement(span);      }    if (!columnsInfo || columnsInfolength === 0) function displayArchitectureSuggestions(result)     }            text:Neural Network Architecture  }  const outputsContainer = documentgetElementById(model-outputs);                        strong>Version:strong> modelInfoproducer_version || Unknown}    html += div>;      let outputShapesHtml = ;      ]      resultsmetricsmae      thtextContent = column;      }  consolelog(CSV columns:columns);      labelObjects = };        trappendChild(td);  if (hasModel && hasData)     html += div>;    });    } finally       }  const headerRow = documentgetElementById(csv-headers);    const byteArray = newUint8Array(byteNumbers);        borderWidth:1      networkfit( animation:duration:500 } });            }    } finally         shuffleData      checkboxtype = checkbox;          columns:-          enabled:true      });        plugins:}      const byteNumbers = newArray(slicelength);                            span class=detail-type>outputdata_type}span>                    span class=info-label>Domain:span>  consolelog(`Switching to tab:tabId}`);  if (filename) filenametextContent = summaryfilename || Unnamed dataset;        }                p>strong>Hidden Layers:strong> resultlayers}p>  const colors = [      if (summaryoutputs)   consolelog(Updating model testing UI);        inputsHtml += `                        strong>outputname}strong>          pointRadius:0    documentgetElementById(metric-mae)textContent = formatNumber(    documentbodyappendChild(a);    );  }let currentPage = 0;            ul>      responsive:true    updateStatus(Load a model first);  documentquerySelectorAll(tab-button)forEach((btn) =>   function renderCsvTable(data)           springLength:120            const nameCell = documentcreateElement(td);                div class=architecture-details>      consoleerror(Model testing error:resulterror);    architectureChart = newChart(ctx       startOptimizationButtonaddEventListener(click startOptimization);    const byteNumbers = newArray(slicelength);    } catch (error)         datasets:[      return;  }        rowappendChild(countCell);    updateStatus(No optimized model available);            text:Actual Values    :0;    if (!modelDetailsElement) return;    } finally       targetSelectappendChild(option);        div class=architecture-card>  let distance = maxDim  2  Mathtan(fov  2);            div class=architecture-card>    byteArrayspush(byteArray);    updateStatus(Model downloaded successfully);    Objectkeys(layers)forEach((key) => delete layers[key]);    } catch (error)     const toPos = positions[edgeto];                inputtype ? `span class=io-type>inputtype}span>` :}        tdtextContent = row[column] !== null ? row[column] :null;      });    if (loadCsvEmptyButton && csvInput)       csvDatarowCount = resultrows;  );  if (emptyState) emptyStatestyledisplay = none;    if (clearDataButton)     }  const height = containerclientHeight || 600;      const nextLayerNodes = [];        if (currentPage > 0)         maintainAspectRatio:false      prevButtondisabled = currentPage === 0;    }      nextPageButtonaddEventListener(click () =>       consoleerror(Error preprocessing data:error);    updateOptimizationUI();  return newBlob(byteArrays  type:mimeType });    tabButtonsforEach((button) => });                div>    showSuccess(resultmessage);    } catch (e)   const pageInfo = documentgetElementById(page-info);                div>  handleTabChange(tab-visualize);    consolelog(Creating 2D network visualization);          pointRadius:5          borderColor:rgba(75 192 192 1)  if (!windowoptimizedModelPath)     container2dclassListremove(active);  }  columnsforEach((column) =>         div class=detail-item>  }      checkboxtype = checkbox;    checkboxvalue = column;      html += p>No input information availablep>;  containerstyleminHeight = 400px;                        inputtype ? `span class=io-type>inputtype}span>` :}  const inputFeaturesSelect = documentgetElementById(input-features);          outputShapesHtml += `function populateFeatureSelectors(columns)   }    prevPageButtondisabled = currentPage === 0;        runModelTest(testSizeSlider);          sortMethod:directed      Objectentries(opTypeCounts)forEach(([opType count] index) =>       }    const text = nodelabel || nodeidtoString();`)
                                })
        }
                                      }
                        })
                })
      })`
                                                                                                                                          }</ul>`</h4></div>`}</h4></td>
</tr>`
                                                                                                                                          }
                                                                                                                                                                                  )
                                                                                                                                                                              })
                                                                                                                                                                              })
                                                                                                                                                                              })
                                                                                                                                                                              }
                                                                                                                                                                              }))
                                                                                                                                                                              })
                                                                                                                                                                              })
                                                                                                                                                                              }}
                                                                                                }
                                                                                                      )
                                                                                              }
                                                                              })</h3>
                                                                                                                                                        }
                                                                                                                                                      })</th></strong>
                                                                                                                                                            }
                                                                                                                                                                                                                                                      })
                                                                                                                                                                                                                                                                                            })
                                                                                                                                                                                                                                            }
                                                                                                                                                                              }
                                                                                                                                            })
                                                                                                                        }
                                                                                                                            })
                                                                                  })
                                                                        })
                                                                      })
                                                                            })
                                                                  })
                                                            ]
                          })}
          })
                                                      }
                                  }
                                  ])
      })
  }