let network = null;
let predictionChart = null;
let architectureChart = null;
let csvData = null;
let currentPage = 0;
let pageSize = 10;
let totalPages = 0;
let threeJsRenderer = null;
let currentViewMode = "2d";
let graphData = null;
let animationFrameId = null;

// document.addEventListener('DOMContentLoaded', function() {
//     console.log('DOM loaded, setting up optimization button');
//
//
//     const startOptBtn = document.getElementById('start-optimization');
//     if (startOptBtn) {
//         console.log('Found optimization button, adding listener');
//         startOptBtn.addEventListener('click', function(e) {
//             console.log('OPTIMIZATION BUTTON CLICKED');
//
//
//
//
//             startOptimization();
//
//
//             e.preventDefault();
//         });
//     } else {
//         console.error('Could not find optimization button!');
//     }
// });

// function initializeOptimizationButton() {
//     console.log('Initializing optimization button');
//
//
//     const startOptBtn = document.getElementById('start-optimization');
//     if (!startOptBtn) {
//         console.error('Optimization button not found');
//         return;
//     }
//
//
//     const newButton = startOptBtn.cloneNode(true);
//     startOptBtn.parentNode.replaceChild(newButton, startOptBtn);
//
//
//     newButton.addEventListener('click', function(e) {
//         console.log('Optimization button clicked');
//
//
//         newButton.disabled = true;
//         newButton.textContent = 'Optimization Running...';
//
//
//         startOptimization().finally(() => {
//
//             newButton.disabled = false;
//             newButton.textContent = 'Start Optimization';
//         });
//
//         e.preventDefault();
//     });
//
//     console.log('Optimization button initialized');
// }

// function updateOptimizationUI() {
//     console.log('Updating optimization UI');
//
//
//     const hasData = window.csvData && window.csvData.data && window.csvData.data.length > 0;
//
//     console.log(`CSV data available: ${hasData}`);
//
//
//     if (hasData) {
//
//         populateOptimizationFeatures();
//     } else {
//         console.log('No CSV data available to populate features');
//     }
// }

// function populateOptimizationFeatures() {
//     console.log('Populating optimization features with auto-select all inputs');
//
//
//     if (!window.csvData || !window.csvData.data || window.csvData.data.length === 0) {
//         console.error('No data available for features');
//         return;
//     }
//
//
//     const inputFeatures = document.getElementById('opt-input-features');
//     const targetFeature = document.getElementById('opt-target-feature');
//
//     if (!inputFeatures || !targetFeature) {
//         console.error('Feature dropdown elements not found');
//         return;
//     }
//
//
//     const columns = Object.keys(window.csvData.data[0]);
//     console.log('CSV columns:', columns);
//
//
//     const inputFeaturesContainer = inputFeatures.parentNode;
//
//
//     const inputFeaturesDisplay = document.createElement('div');
//     inputFeaturesDisplay.id = 'input-features-display';
//     inputFeaturesDisplay.style.border = '1px solid #d1d5db';
//     inputFeaturesDisplay.style.borderRadius = '6px';
//     inputFeaturesDisplay.style.padding = '8px';
//     inputFeaturesDisplay.style.backgroundColor = '#f9fafb';
//     inputFeaturesDisplay.style.minHeight = '100px';
//     inputFeaturesDisplay.style.maxHeight = '150px';
//     inputFeaturesDisplay.style.overflowY = 'auto';
//
//
//     inputFeatures.style.display = 'none';
//
//
//     inputFeatures.innerHTML = '';
//
//
//     targetFeature.innerHTML = '<option value="" disabled selected>-- Select Target Feature --</option>';
//
//
//     columns.forEach(column => {
//
//         const inputOption = document.createElement('option');
//         inputOption.value = column;
//         inputOption.textContent = column;
//         inputOption.selected = true;
//         inputFeatures.appendChild(inputOption);
//
//
//         const featureItem = document.createElement('div');
//         featureItem.style.marginBottom = '4px';
//         featureItem.innerHTML = `<span style="display:inline-block; width:8px; height:8px; background-color:#4f46e5; border-radius:50%; margin-right:6px;"></span> ${column}`;
//         inputFeaturesDisplay.appendChild(featureItem);
//
//
//         const targetOption = document.createElement('option');
//         targetOption.value = column;
//         targetOption.textContent = column;
//         targetFeature.appendChild(targetOption);
//     });
//
//
//     const helperText = document.createElement('div');
//     helperText.innerHTML = '<small style="color: #6b7280; margin-top: 4px; display: block;">All columns will be used as input features automatically (except target).</small>';
//
//
//     inputFeaturesContainer.insertBefore(inputFeaturesDisplay, inputFeatures);
//     inputFeaturesContainer.appendChild(helperText);
//
//     console.log('Feature dropdowns populated successfully');
// }

async function startOptimization() {
  console.log("OPTIMIZATION STARTED");

  const inputFeatures = document.getElementById("opt-input-features");
  const targetFeature = document.getElementById("opt-target-feature");
  const maxEpochs = document.getElementById("opt-max-epochs");
  const progressElement = document.getElementById("optimization-progress");
  const resultsElement = document.getElementById("optimization-results");

  if (!inputFeatures || !targetFeature || !maxEpochs) {
    console.error("Missing form elements");

    return;
  }

  const allInputs = Array.from(inputFeatures.options).map((opt) => opt.value);

  const selectedTarget = targetFeature.value;
  const epochs = parseInt(maxEpochs.value) || 10;

  if (!selectedTarget) {
    return;
  }

  const selectedInputs = allInputs.filter((input) => input !== selectedTarget);

  console.log("Selected inputs:", selectedInputs);
  console.log("Selected target:", selectedTarget);
  console.log("Max epochs:", epochs);

  if (progressElement) progressElement.style.display = "block";
  if (resultsElement) resultsElement.style.display = "none";

  const progressBar = document.getElementById("opt-progress-bar");
  const progressText = document.getElementById("opt-progress-text");

  if (progressBar) progressBar.style.width = "0%";
  if (progressText) progressText.textContent = "Starting optimization...";

  console.log("Calling Python function directly...");

  try {
    console.log("Calling Python function directly...");

    const result = await eel.find_optimal_architecture(
      selectedInputs,
      selectedTarget,
      epochs,
    )();

    console.log("Optimization completed, result:", result);

    displayOptimizationResults(result);

    return result;
  } catch (error) {
    console.error("Error during optimization:", error);
    throw error;
  } finally {
    showSpinner(false);
  }
}

function updateOptimizationProgress(progress) {
  console.log("Progress update received:", progress);

  const progressBar = document.getElementById("opt-progress-bar");
  const progressText = document.getElementById("opt-progress-text");

  if (!progressBar || !progressText) {
    console.error("Progress elements not found");
    return;
  }

  if (typeof progress.progress === "number") {
    progressBar.style.width = `${progress.progress}%`;

    progressBar.setAttribute("aria-valuenow", progress.progress);
  }

  if (progress.status) {
    progressText.textContent = progress.status;
  }

  if (progress.error) {
    progressBar.classList.add("bg-danger");
    console.error("Optimization error:", progress.status);
  }
}

function displayOptimizationResults(results) {
  const progressElement = document.getElementById("optimization-progress");
  const resultsElement = document.getElementById("optimization-results");

  if (progressElement) progressElement.style.display = "none";
  if (resultsElement) resultsElement.style.display = "block";

  const bestConfig = results.best_config;

  if (!bestConfig) {
    updateStatus("No best configuration found");
    return;
  }

  document.getElementById("best-layers").textContent = bestConfig.layers;
  document.getElementById("best-neurons").textContent = bestConfig.neurons;
  document.getElementById("best-loss").textContent =
    bestConfig.test_loss.toFixed(6);
  document.getElementById("best-r2").textContent =
    bestConfig.r2_score.toFixed(4);
}

function createArchitectureComparisonChart(results) {
  const ctx = document
    .getElementById("architecture-comparison-chart")
    .getContext("2d");

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

  const chart = new Chart(ctx, {
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
              const item =
                dataByLayers[context.dataset.label.split(" ")[0]][
                  context.dataIndex
                ];
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

async function downloadOptimizedModel() {
  if (!window.optimizedModelPath) {
    updateStatus("No optimized model available");
    return;
  }

  try {
    showSpinner(true);
    updateStatus("Preparing model for download...");

    const result = await window.eel.download_optimized_model(
      window.optimizedModelPath,
    )();

    if (result.error) {
      updateStatus(`Error: ${result.error}`);
      return;
    }

    const byteCharacters = atob(result.base64);
    const byteArrays = [];

    for (let offset = 0; offset < byteCharacters.length; offset += 512) {
      const slice = byteCharacters.slice(offset, offset + 512);

      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }

      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }

    const blob = new Blob(byteArrays, { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = result.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
    updateStatus("Model downloaded successfully");
  } catch (error) {
    console.error("Error downloading model:", error);
    updateStatus(`Error: ${error.message}`);
  } finally {
    showSpinner(false);
  }
}

function base64ToBlob(base64, mimeType) {
  const byteCharacters = atob(base64);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);

    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  return new Blob(byteArrays, { type: mimeType });
}

function populateOptimizationFeatures() {
  console.log("Populating optimization features");

  if (
    !window.csvData ||
    !window.csvData.data ||
    window.csvData.data.length === 0
  ) {
    console.error("No data available for features");
    return;
  }

  const inputFeatures = document.getElementById("opt-input-features");
  const targetFeature = document.getElementById("opt-target-feature");

  if (!inputFeatures || !targetFeature) {
    console.error("Feature dropdown elements not found");
    return;
  }

  while (inputFeatures.options.length > 0) {
    inputFeatures.remove(0);
  }

  targetFeature.innerHTML =
    '<option value="" disabled selected>-- Select Target Feature --</option>';

  inputFeatures.setAttribute("multiple", "multiple");
  inputFeatures.setAttribute("size", "5");

  const columns = Object.keys(window.csvData.data[0]);
  console.log("CSV columns:", columns);

  columns.forEach((column) => {
    const inputOption = document.createElement("option");
    inputOption.value = column;
    inputOption.textContent = column;
    inputFeatures.appendChild(inputOption);

    const targetOption = document.createElement("option");
    targetOption.value = column;
    targetOption.textContent = column;
    targetFeature.appendChild(targetOption);
  });
}

async function runModelTest(testSizeSlider) {
  const inputFeaturesSelect = document.getElementById("input-features");
  const targetFeatureSelect = document.getElementById("target-feature");

  if (!testSizeSlider) {
    testSizeSlider = document.getElementById("test-size");
  }

  if (!inputFeaturesSelect || !targetFeatureSelect) {
    updateStatus("Feature selectors not found");
    return;
  }

  const inputFeatures = Array.from(inputFeaturesSelect.selectedOptions).map(
    (option) => option.value,
  );

  const targetFeature = targetFeatureSelect.value;

  const testSize = parseFloat(testSizeSlider.value);

  if (inputFeatures.length === 0) {
    updateStatus("Please select at least one input feature");
    return;
  }

  if (!targetFeature) {
    updateStatus("Please select a target feature");
    return;
  }

  showSpinner(true);
  updateStatus("Running model test...");

  try {
    const result = await window.eel.test_model(
      inputFeatures,
      targetFeature,
      testSize,
    )();

    if (result.error) {
      updateStatus(`Error: ${result.error}`);
      console.error("Model testing error:", result.error);
      return;
    }

    displayTestResults(result);
    updateStatus("Model test completed successfully");
  } catch (error) {
    console.error("Error running model test:", error);
    updateStatus(`Error: ${error.message}`);
  } finally {
    showSpinner(false);
  }
}

function displayTestResults(results) {
  const resultsContainer = document.getElementById("testing-results");
  if (resultsContainer) {
    resultsContainer.style.display = "block";
  }

  if (results.metrics) {
    const formatNumber = (num) => (Number.isFinite(num) ? num.toFixed(4) : "-");

    document.getElementById("metric-mse").textContent = formatNumber(
      results.metrics.mse,
    );
    document.getElementById("metric-mae").textContent = formatNumber(
      results.metrics.mae,
    );
    document.getElementById("metric-r2").textContent = formatNumber(
      results.metrics.r2,
    );
    document.getElementById("metric-samples").textContent =
      results.metrics.samples;
  }

  if (results.results && results.results.length > 0) {
    createPredictionChart(results.results);
  }
}

function createPredictionChart(results) {
  const ctx = document.getElementById("predictions-chart").getContext("2d");

  const actual = results.map((item) => item.actual);
  const predicted = results.map((item) => item.predicted);

  const allValues = [...actual, ...predicted];
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  if (predictionChart) {
    predictionChart.destroy();
  }

  predictionChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Predictions",
          data: results.map((item) => ({ x: item.actual, y: item.predicted })),
          backgroundColor: "rgba(75, 192, 192, 0.6)",
          borderColor: "rgba(75, 192, 192, 1)",
          borderWidth: 1,
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: "Perfect Prediction",
          data: [
            { x: min, y: min },
            { x: max, y: max },
          ],
          type: "line",
          borderColor: "rgba(255, 99, 132, 0.7)",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      aspectRatio: 1.5,
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              return `Actual: ${context.parsed.x.toFixed(4)}, Predicted: ${context.parsed.y.toFixed(4)}`;
            },
          },
        },
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: "Predicted vs Actual Values",
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Actual Values",
          },
        },
        y: {
          title: {
            display: true,
            text: "Predicted Values",
          },
        },
      },
    },
  });
}

function updateModelTestingUI() {
  console.log("Updating model testing UI...");

  const hasModel = window.graphData && window.graphData.summary;
  const hasData =
    window.csvData && window.csvData.data && window.csvData.data.length > 0;

  console.log(`Model available: ${hasModel}, Data available: ${hasData}`);

  const emptyState = document.getElementById("testing-empty-state");
  const testingContent = document.getElementById("testing-content");
  const runTestButton = document.getElementById("run-test");

  console.log(`Found UI elements: 
        Empty state: ${!!emptyState}
        Testing content: ${!!testingContent}
        Run button: ${!!runTestButton}
    `);

  if (hasModel && hasData) {
    if (emptyState) emptyState.style.display = "none";
    if (testingContent) testingContent.style.display = "block";
    if (runTestButton) runTestButton.disabled = false;

    console.log("Showing testing UI and populating feature dropdowns");

    populateFeatureDropdowns();
  } else {
    if (emptyState) emptyState.style.display = "flex";
    if (testingContent) testingContent.style.display = "none";
    if (runTestButton) runTestButton.disabled = true;

    console.log("Showing empty state message");
  }
}

function displayModelSummary(summary) {
  console.log("Displaying model summary:", summary);

  const modelInfoContainer = document.getElementById("model-info");
  const layersTableBody = document.querySelector("#layers-table tbody");
  const inputsContainer = document.getElementById("model-inputs");
  const outputsContainer = document.getElementById("model-outputs");

  if (modelInfoContainer) {
    modelInfoContainer.innerHTML = `
            <div class="info-card">
                <div class="info-row">
                    <span class="info-label">Producer:</span>
                    <span class="info-value">${summary.producer || "Unknown"}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">IR Version:</span>
                    <span class="info-value">${summary.ir_version || "Unknown"}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Domain:</span>
                    <span class="info-value">${summary.domain || "General"}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Node Count:</span>
                    <span class="info-value">${summary.node_count || summary.nodes?.length || 0}</span>
                </div>
            </div>
        `;
  }

  if (inputsContainer) {
    let inputsHtml = "<h4>Inputs</h4>";
    if (summary.inputs && summary.inputs.length > 0) {
      inputsHtml += '<ul class="model-io-list">';
      summary.inputs.forEach((input) => {
        inputsHtml += `
                    <li class="model-io-item">
                        <strong>${input.name}</strong>
                        ${input.type ? `<span class="io-type">${input.type}</span>` : ""}
                    </li>
                `;
      });
      inputsHtml += "</ul>";
    } else {
      inputsHtml += "<p>No input information available</p>";
    }
    inputsContainer.innerHTML = inputsHtml;
  }

  if (outputsContainer) {
    let outputsHtml = "<h4>Outputs</h4>";
    if (summary.outputs && summary.outputs.length > 0) {
      outputsHtml += '<ul class="model-io-list">';
      summary.outputs.forEach((output) => {
        outputsHtml += `
                    <li class="model-io-item">
                        <strong>${output.name}</strong>
                        ${output.type ? `<span class="io-type">${output.type}</span>` : ""}
                    </li>
                `;
      });
      outputsHtml += "</ul>";
    } else {
      outputsHtml += "<p>No output information available</p>";
    }
    outputsContainer.innerHTML = outputsHtml;
  }

  if (layersTableBody) {
    layersTableBody.innerHTML = "";

    if (summary.nodes && summary.nodes.length > 0) {
      const opTypeCounts = {};

      summary.nodes.forEach((node) => {
        const opType = node.op_type || "Unknown";
        opTypeCounts[opType] = (opTypeCounts[opType] || 0) + 1;
      });

      Object.entries(opTypeCounts).forEach(([opType, count], index) => {
        const row = document.createElement("tr");

        const indexCell = document.createElement("td");
        indexCell.textContent = index + 1;
        row.appendChild(indexCell);

        const typeCell = document.createElement("td");
        typeCell.textContent = opType;
        row.appendChild(typeCell);

        const countCell = document.createElement("td");
        countCell.textContent = `${count} node${count > 1 ? "s" : ""}`;
        row.appendChild(countCell);

        layersTableBody.appendChild(row);
      });

      const totalRow = document.createElement("tr");
      totalRow.classList.add("total-row");

      const totalLabelCell = document.createElement("td");
      totalLabelCell.colSpan = 2;
      totalLabelCell.textContent = "Total Operations";
      totalLabelCell.style.fontWeight = "bold";
      totalRow.appendChild(totalLabelCell);

      const totalCountCell = document.createElement("td");
      totalCountCell.textContent = `${summary.nodes.length} nodes`;
      totalCountCell.style.fontWeight = "bold";
      totalRow.appendChild(totalCountCell);

      layersTableBody.appendChild(totalRow);
    } else {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 3;
      cell.textContent = "No layer information available";
      row.appendChild(cell);
      layersTableBody.appendChild(row);
    }
  }
}

function displayCsvData(csvData, page = 0) {
  if (!csvData || !csvData.data || csvData.data.length === 0) {
    console.log("No data to display");
    return;
  }

  const pageSize = 10;
  const totalPages = Math.ceil(csvData.data.length / pageSize);
  const startIndex = page * pageSize;
  const endIndex = Math.min(startIndex + pageSize, csvData.data.length);

  const tableView = document.getElementById("data-table-view");
  const emptyState = document.getElementById("data-empty-state");
  const headerRow = document.getElementById("csv-headers");
  const tableBody = document.getElementById("csv-body");

  if (tableView) tableView.style.display = "block";
  if (emptyState) emptyState.style.display = "none";

  if (headerRow) headerRow.innerHTML = "";
  if (tableBody) tableBody.innerHTML = "";

  const columns = Object.keys(csvData.data[0]);

  if (headerRow) {
    columns.forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column;
      headerRow.appendChild(th);
    });
  }

  if (tableBody) {
    const pageData = csvData.data.slice(startIndex, endIndex);

    pageData.forEach((row) => {
      const tr = document.createElement("tr");

      columns.forEach((column) => {
        const td = document.createElement("td");
        td.textContent = row[column] !== null ? row[column] : "null";
        tr.appendChild(td);
      });

      tableBody.appendChild(tr);
    });
  }

  updatePaginationControls(page, totalPages);
}

function updatePaginationControls(currentPage, totalPages) {
  const prevPageButton = document.getElementById("prev-page");
  const nextPageButton = document.getElementById("next-page");
  const pageInfo = document.getElementById("page-info");

  console.log(`Updating pagination: page ${currentPage + 1} of ${totalPages}`);

  if (prevPageButton) {
    prevPageButton.disabled = currentPage === 0;
  }

  if (nextPageButton) {
    nextPageButton.disabled = currentPage >= totalPages - 1;
  }

  if (pageInfo) {
    pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
  }
}

function updateDataSummary(summary) {
  if (!summary) return;

  const rowCount = document.getElementById("data-row-count");
  const columnCount = document.getElementById("data-column-count");
  const filename = document.getElementById("data-filename");
  const missingCount = document.getElementById("data-missing-count");

  if (rowCount) rowCount.textContent = summary.rows || "-";
  if (columnCount) columnCount.textContent = summary.columns || "-";
  if (filename) filename.textContent = summary.filename || "Unnamed dataset";

  const totalMissing = summary.missing_values
    ? Object.values(summary.missing_values).reduce((sum, val) => sum + val, 0)
    : 0;

  if (missingCount) missingCount.textContent = totalMissing;
}

function updatePaginationInfo(currentPage, totalPages) {
  const pageInfo = document.getElementById("page-info");
  const prevButton = document.getElementById("prev-page");
  const nextButton = document.getElementById("next-page");

  if (pageInfo)
    pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;

  if (prevButton) prevButton.disabled = currentPage <= 0;
  if (nextButton) nextButton.disabled = currentPage >= totalPages - 1;
}

function handleTabChange(tabId) {
  console.log(`Switching to tab: ${tabId}`);

  document.querySelectorAll(".tab-button").forEach((btn) => {
    btn.classList.remove("active");
  });

  const activeButton = document.getElementById(tabId);
  if (activeButton) {
    activeButton.classList.add("active");
  }

  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.remove("active");

    pane.style.display = "none";

    console.log(`Hidden tab: ${pane.id}`);
  });

  const contentId = "content-" + tabId.replace("tab-", "");
  const contentPane = document.getElementById(contentId);

  if (contentPane) {
    contentPane.classList.add("active");

    contentPane.style.display = "block";

    console.log(`Showing tab: ${contentId}`);
  } else {
    console.error(`Tab content with ID ${contentId} not found`);
  }
}

function updateStatus(message) {
  const status = document.getElementById("status");
  if (status) {
    status.textContent = message;
  }
  console.log("Status:", message);
}

function createNetworkVisualization(graphData) {
  try {
    console.log("Creating 2D network visualization");

    window.graphData = graphData;

    const container = document.getElementById("network-2d");
    if (!container) {
      console.error("2D network container not found");
      return;
    }

    container.innerHTML = "";

    if (!graphData || !graphData.nodes || !graphData.edges) {
      container.innerHTML =
        '<div class="error-message">No valid graph data available</div>';
      return;
    }

    console.log(
      `Creating 2D visualization with ${graphData.nodes.length} nodes`,
    );

    const nodes = new vis.DataSet(graphData.nodes);
    const edges = new vis.DataSet(graphData.edges);

    const options = {
      layout: {
        hierarchical: {
          enabled: true,
          direction: "LR",
          sortMethod: "directed",
          levelSeparation: 150,
          nodeSpacing: 120,
        },
      },
      nodes: {
        shape: "box",
        margin: 10,
        font: { size: 14, face: "Inter" },
        borderWidth: 1,
        shadow: true,
      },
      edges: {
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        smooth: { type: "cubicBezier", roundness: 0.5 },
      },
      physics: {
        enabled: true,
        stabilization: {
          enabled: true,
          iterations: 1000,
        },
        hierarchicalRepulsion: {
          centralGravity: 0.0,
          springLength: 120,
          springConstant: 0.01,
          nodeDistance: 150,
        },
        solver: "hierarchicalRepulsion",
      },
    };

    network = new vis.Network(container, { nodes, edges }, options);

    network.once("stabilizationIterationsDone", () => {
      network.fit({ animation: { duration: 500 } });
      console.log("Network stabilized");
    });
  } catch (error) {
    console.error("Error creating 2D network:", error);
    updateStatus(`Error: ${error.message}`);
  }
}

function switchToMode(mode) {
  console.log(`Switching to ${mode} mode`);

  if (!window.graphData) {
    console.log("No graph data available");
    updateStatus("Load a model first");
    return;
  }

  const view2dButton = document.getElementById("view-2d");
  const view3dButton = document.getElementById("view-3d");

  if (view2dButton) view2dButton.classList.toggle("active", mode === "2d");
  if (view3dButton) view3dButton.classList.toggle("active", mode === "3d");

  const container2d = document.getElementById("network-2d");
  const container3d = document.getElementById("network-3d");

  if (!container2d || !container3d) {
    console.error("Missing containers");
    return;
  }

  currentViewMode = mode;

  if (mode === "2d") {
    container2d.style.display = "block";
    container3d.style.display = "none";
    container2d.classList.add("active");
    container3d.classList.remove("active");
  } else {
    container2d.style.display = "none";
    container3d.style.display = "block";
    container2d.classList.remove("active");
    container3d.classList.add("active");
  }

  if (mode === "2d" && animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (mode === "2d") {
    if (!network) {
      createNetworkVisualization(window.graphData);
    } else {
      setTimeout(() => {
        network.redraw();
        network.fit();
      }, 100);
    }
  } else {
    container3d.innerHTML = "";
    nodeObjects = {};
    labelObjects = {};

    container3d.style.width = "100%";
    container3d.style.height = "100%";

    create3DNetworkVisualization(window.graphData);
  }

  updateStatus(`Showing ${mode.toUpperCase()} visualization`);
}

function animate() {
  if (!renderer || !scene || !camera) return;

  animationFrameId = requestAnimationFrame(animate);

  if (controls) controls.update();

  updateLabelsPositions();

  renderer.render(scene, camera);
}

let scene, camera, renderer, controls;
let nodeObjects = {};
let labelObjects = {};

function create3DNetworkVisualization(data) {
  console.log("Creating 3D network visualization");

  const container = document.getElementById("network-3d");
  if (!container) {
    console.error("3D network container not found");
    return;
  }

  window.graphData = data;

  const canvasElements = container.querySelectorAll("canvas");
  canvasElements.forEach((canvas) => canvas.remove());

  const labelsContainer = document.getElementById("three-labels-container");
  if (labelsContainer) {
    labelsContainer.remove();
  }

  if (renderer) {
    try {
      renderer.dispose();
      scene = null;
      camera = null;
      controls = null;
      nodeObjects = {};
      labelObjects = {};
    } catch (e) {
      console.error("Error cleaning up THREE.js:", e);
    }
  }

  setupThreeJsScene(container);

  const nodePositions = calculateNodePositions(data.nodes, data.edges);

  createNodes(data.nodes, nodePositions);
  createEdges(data.edges, nodePositions);
  createLabels(data.nodes, nodePositions);

  setTimeout(fitCameraToGraph, 100);

  animate();
}

function fitCameraToGraph() {
  if (!scene || !camera) return;

  const nodePositions = [];
  Object.values(nodeObjects).forEach((node) => {
    nodePositions.push(node.position);
  });

  if (nodePositions.length === 0) return;

  const box = new THREE.Box3().setFromPoints(nodePositions);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z);
  const fov = camera.fov * (Math.PI / 180);
  let distance = maxDim / 2 / Math.tan(fov / 2);

  distance *= 1.5;

  camera.position.set(center.x, center.y, center.z + distance);

  controls.target.copy(center);
  controls.update();
}

function setupThreeJsScene(container) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;
  console.log(`Container dimensions: ${width}x${height}`);

  const aspect = width / height;

  camera = new THREE.PerspectiveCamera(40, aspect, 1, 10000);
  camera.position.set(0, 0, 2000);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);

  renderer.domElement.style.position = "absolute";
  renderer.domElement.style.top = "40px";
  renderer.domElement.style.left = "0";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "calc(100% - 40px)";
  renderer.domElement.style.zIndex = "1";

  container.style.display = "block";
  container.style.position = "relative";
  container.style.minHeight = "400px";

  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed = 0.5;
  controls.panSpeed = 0.8;
  controls.zoomSpeed = 1.0;
  controls.minDistance = 100;
  controls.maxDistance = 5000;
  controls.target.set(0, 0, 0);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 1).normalize();
  scene.add(directionalLight);

  const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
  backLight.position.set(-1, -1, -1).normalize();
  scene.add(backLight);
}

function calculateNodePositions(nodes, edges) {
  const layers = {};
  const nodeMap = {};

  nodes.forEach((node) => {
    nodeMap[node.id] = node;

    if (node.layer === undefined) {
      node.layer = 0;
    }

    if (!layers[node.layer]) {
      layers[node.layer] = [];
    }

    layers[node.layer].push(node);
  });

  if (Object.keys(layers).length <= 1) {
    const incomingEdges = {};
    edges.forEach((edge) => {
      if (!incomingEdges[edge.to]) {
        incomingEdges[edge.to] = 0;
      }
      incomingEdges[edge.to]++;
    });

    Object.keys(layers).forEach((key) => delete layers[key]);

    let currentLayer = 0;
    const assignedNodes = new Set();
    const nodesToProcess = nodes
      .filter((node) => !incomingEdges[node.id])
      .map((node) => node.id);

    while (nodesToProcess.length > 0) {
      layers[currentLayer] = [];

      const nextLayerNodes = [];

      nodesToProcess.forEach((nodeId) => {
        assignedNodes.add(nodeId);
        const node = nodeMap[nodeId];
        node.layer = currentLayer;
        layers[currentLayer].push(node);

        edges.forEach((edge) => {
          if (edge.from === nodeId && !assignedNodes.has(edge.to)) {
            nextLayerNodes.push(edge.to);
          }
        });
      });

      currentLayer++;
      nodesToProcess.length = 0;
      nodesToProcess.push(...nextLayerNodes);
    }
  }

  const positions = {};
  const layerKeys = Object.keys(layers)
    .map(Number)
    .sort((a, b) => a - b);
  const layerSpacing = 200;
  const nodeSpacing = 120;

  layerKeys.forEach((layerIndex) => {
    const nodesInLayer = layers[layerIndex];
    const layerX = (layerIndex - layerKeys.length / 2) * layerSpacing;

    nodesInLayer.forEach((node, i) => {
      const offsetY = (i - (nodesInLayer.length - 1) / 2) * nodeSpacing;
      positions[node.id] = {
        x: layerX,
        y: offsetY,
        z: 0,
      };
    });
  });

  return positions;
}

function createNodes(nodes, positions) {
  const nodeTypes = {};

  nodes.forEach((node) => {
    const type = node.type || "unknown";
    if (!nodeTypes[type]) {
      nodeTypes[type] = 0;
    }
    nodeTypes[type]++;
  });

  const typeColors = {};
  const colors = [
    0x4f46e5, 0x7c3aed, 0xe11d48, 0xf59e0b, 0x10b981, 0x3b82f6, 0xec4899,
    0x8b5cf6,
  ];

  Object.keys(nodeTypes).forEach((type, i) => {
    typeColors[type] = colors[i % colors.length];
  });

  nodes.forEach((node) => {
    if (!positions[node.id]) {
      console.warn(`No position found for node ${node.id}`);
      return;
    }

    const pos = positions[node.id];
    const size = node.size || 15;
    const color = node.color
      ? parseInt(node.color.replace("#", "0x"))
      : typeColors[node.type || "unknown"] || 0x4f46e5;

    const geometry = new THREE.SphereGeometry(size, 16, 16);
    const material = new THREE.MeshLambertMaterial({
      color: color,
      emissive: 0x222222,
      emissiveIntensity: 0.2,
      transparent: true,
      opacity: 0.9,
    });

    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(pos.x, pos.y, pos.z);
    sphere.userData = { id: node.id, label: node.label, type: node.type };

    scene.add(sphere);
    nodeObjects[node.id] = sphere;
  });
}

function createEdges(edges, positions) {
  edges.forEach((edge) => {
    if (!positions[edge.from] || !positions[edge.to]) {
      console.warn(`Missing positions for edge ${edge.from} -> ${edge.to}`);
      return;
    }

    const fromPos = positions[edge.from];
    const toPos = positions[edge.to];

    const points = [];
    points.push(new THREE.Vector3(fromPos.x, fromPos.y, fromPos.z));
    points.push(new THREE.Vector3(toPos.x, toPos.y, toPos.z));

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0x9ca3af,
      linewidth: 1,
    });

    const line = new THREE.Line(geometry, material);
    scene.add(line);
  });
}

function createLabels(nodes, positions) {
  let labelsContainer = document.getElementById("three-labels-container");
  if (labelsContainer) {
    labelsContainer.remove();
  }

  labelsContainer = document.createElement("div");
  labelsContainer.id = "three-labels-container";
  labelsContainer.style.position = "absolute";
  labelsContainer.style.top = "0";
  labelsContainer.style.left = "0";
  labelsContainer.style.width = "100%";
  labelsContainer.style.height = "100%";
  labelsContainer.style.pointerEvents = "none";
  labelsContainer.style.zIndex = "10";

  const container3d = document.getElementById("network-3d");
  container3d.appendChild(labelsContainer);

  labelObjects = {};

  nodes.forEach((node) => {
    if (!positions[node.id]) return;

    const text = node.label || node.id.toString();

    const labelDiv = document.createElement("div");
    labelDiv.className = "node-label";
    labelDiv.textContent = text;

    labelsContainer.appendChild(labelDiv);
    labelObjects[node.id] = labelDiv;
  });
}

function updateLabelsPositions() {
  if (!camera || !renderer || !scene) return;

  const rendererRect = renderer.domElement.getBoundingClientRect();

  Object.keys(nodeObjects).forEach((nodeId) => {
    const nodeMesh = nodeObjects[nodeId];
    const label = labelObjects[nodeId];

    if (!nodeMesh || !label) return;

    const worldPos = nodeMesh.position.clone();

    worldPos.project(camera);

    const x = ((worldPos.x + 1) / 2) * rendererRect.width;
    const y = ((-worldPos.y + 1) / 2) * rendererRect.height;

    const distance = camera.position.distanceTo(nodeMesh.position);

    label.style.left = `${x}px`;
    label.style.top = `${y}px`;
    label.style.transform = "translate(-50%, -50%)";

    if (worldPos.z > 1 || worldPos.z < -1) {
      label.style.display = "none";
    } else {
      label.style.display = "block";
      const opacity = Math.max(0.3, 1 - distance / 1500);
      label.style.opacity = opacity.toFixed(2);

      const zIndex = Math.floor(1000 - distance);
      label.style.zIndex = zIndex;
    }
  });
}

document.addEventListener("DOMContentLoaded", function () {
  console.log("DOM loaded, initializing application...");

  handleTabChange("tab-visualize");
  updateOptimizationUI();
  updateModelTestingUI();

  function waitForEel() {
    if (typeof window.eel === "undefined") {
      console.log("Waiting for Eel to initialize...");
      setTimeout(waitForEel, 100);
      return;
    }

    console.log("Eel initialized successfully");
    setupEventListeners();
  }

  waitForEel();

  function setupEventListeners() {
    console.log("Setting up event listeners...");

    const tabButtons = document.querySelectorAll(".tab-button");
    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const tabId = button.id;
        console.log(`Tab button clicked: ${tabId}`);
        handleTabChange(tabId);
      });
    });

    const modelInput = document.getElementById("model-input");
    if (modelInput) {
      modelInput.addEventListener("change", handleModelUpload);
      console.log("Model input event handler added");
    }

    const csvInput = document.getElementById("csv-input");
    const loadCsvButton = document.getElementById("btn-load-csv");
    const loadCsvEmptyButton = document.getElementById("btn-load-csv-empty");
    const clearDataButton = document.getElementById("btn-clear-data");
    const preprocessButton = document.getElementById("btn-preprocess");

    if (loadCsvButton && csvInput) {
      loadCsvButton.addEventListener("click", () => {
        csvInput.click();
      });
    }

    if (loadCsvEmptyButton && csvInput) {
      loadCsvEmptyButton.addEventListener("click", () => {
        csvInput.click();
      });
    }

    if (csvInput) {
      csvInput.addEventListener("change", handleCsvUpload);
    }

    if (clearDataButton) {
      clearDataButton.addEventListener("click", function () {
        console.log("Clear data button clicked");

        window.csvData = null;

        currentPage = 0;

        const tableView = document.getElementById("data-table-view");
        const emptyState = document.getElementById("data-empty-state");

        if (tableView) tableView.style.display = "none";
        if (emptyState) emptyState.style.display = "flex";

        updateDataSummary({
          rows: "-",
          columns: "-",
          filename: "No file loaded",
          missing_values: {},
        });

        if (preprocessButton) preprocessButton.disabled = true;
        if (clearDataButton) clearDataButton.disabled = true;

        updateStatus("Data cleared");
      });
    }

    const prevPageButton = document.getElementById("prev-page");
    const nextPageButton = document.getElementById("next-page");

    if (prevPageButton) {
      prevPageButton.addEventListener("click", () => {
        console.log("Previous page button clicked");
        if (currentPage > 0) {
          currentPage--;
          console.log(`Moving to page ${currentPage + 1}`);
          displayCsvData(window.csvData, currentPage);
        }
      });
    }

    if (nextPageButton) {
      nextPageButton.addEventListener("click", () => {
        console.log("Next page button clicked");
        if (window.csvData && window.csvData.data) {
          const pageSize = 10;
          const totalPages = Math.ceil(window.csvData.data.length / pageSize);
          if (currentPage < totalPages - 1) {
            currentPage++;
            console.log(`Moving to page ${currentPage + 1}`);
            displayCsvData(window.csvData, currentPage);
          }
        }
      });
    }

    const testSizeSlider = document.getElementById("test-size");
    const testSizeValue = document.getElementById("test-size-value");
    const runTestButton = document.getElementById("run-test");

    if (testSizeSlider && testSizeValue) {
      testSizeSlider.addEventListener("input", () => {
        const value = parseFloat(testSizeSlider.value);
        testSizeValue.textContent = `${Math.round(value * 100)}%`;
      });
    }

    if (runTestButton) {
      runTestButton.addEventListener("click", () => {
        const testSizeSlider = document.getElementById("test-size");
        runModelTest(testSizeSlider);
      });
    }

    const view2dButton = document.getElementById("view-2d");
    const view3dButton = document.getElementById("view-3d");

    if (view2dButton) {
      view2dButton.addEventListener("click", () => {
        console.log("Switching to 2D mode");
        switchToMode("2d");
      });
    }

    if (view3dButton) {
      view3dButton.addEventListener("click", () => {
        console.log("Switching to 3D mode");
        switchToMode("3d");
      });
    }

    console.log("Event listeners set up successfully");

    const startOptimizationButton =
      document.getElementById("start-optimization");
    const downloadModelButton = document.getElementById("download-model");

    if (startOptimizationButton) {
      startOptimizationButton.addEventListener("click", startOptimization);
      console.log("Start optimization button event listener added");
    }

    if (downloadModelButton) {
      downloadModelButton.addEventListener("click", downloadOptimizedModel);
      console.log("Download model button event listener added");
    }

    updateOptimizationUI();
  }

  async function handleModelUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
      updateStatus(`Loading ${file.name}...`);
      showSpinner(true);

      const arrayBuffer = await readFileAsArrayBuffer(file);
      const base64 = arrayBufferToBase64(arrayBuffer);

      const result = await window.eel.load_onnx_model(base64)();

      if (result.error) {
        updateStatus(`Error: ${result.error}`);
        return;
      }

      window.graphData = result;

      updateStatus(`Model loaded successfully: ${file.name}`);

      network = null;
      scene = null;
      camera = null;
      renderer = null;
      controls = null;

      switchToMode(currentViewMode);

      updateModelDetails(result.summary);

      displayModelSummary(result.summary);

      if (result.summary && result.summary.nodes) {
        const layersTableBody = document.querySelector("#layers-table tbody");
        if (layersTableBody) {
          layersTableBody.innerHTML = "";

          result.summary.nodes.forEach((node, index) => {
            const row = document.createElement("tr");

            const indexCell = document.createElement("td");
            indexCell.textContent = index + 1;
            row.appendChild(indexCell);

            const nameCell = document.createElement("td");
            nameCell.textContent = node.name || "Unnamed";
            row.appendChild(nameCell);

            const typeCell = document.createElement("td");
            typeCell.textContent = node.op_type || "Unknown";
            row.appendChild(typeCell);

            layersTableBody.appendChild(row);
          });
        }
      }
    } catch (error) {
      console.error("Error loading model:", error);
      updateStatus(`Error: ${error.message}`);
    } finally {
      showSpinner(false);
    }
    if (window.csvData) {
      updateModelTestingUI();
      updateOptimizationUI();
    }

    if (window.graphData) {
      updateModelTestingUI();
      updateOptimizationUI();
    }
    updateOptimizationUI();
  }

  function updateModelDetails(summary) {
    const modelDetailsElement = document.getElementById("model-details");
    if (!modelDetailsElement) return;

    if (!summary) {
      modelDetailsElement.innerHTML = "No model details available";
      return;
    }

    const opTypeCounts = {};
    if (summary.nodes && summary.nodes.length > 0) {
      summary.nodes.forEach((node) => {
        const opType = node.op_type || "Unknown";
        opTypeCounts[opType] = (opTypeCounts[opType] || 0) + 1;
      });
    }

    let html = '<div class="model-details-content">';

    html += `<div class="detail-section">
        <h4>Model Information</h4>
        <div class="detail-item">
            <strong>Producer:</strong> ${summary.producer || "Unknown"}
        </div>
        <div class="detail-item">
            <strong>IR Version:</strong> ${summary.ir_version || "Unknown"}
        </div>
        <div class="detail-item">
            <strong>Domain:</strong> ${summary.domain || "General"}
        </div>
        <div class="detail-item">
            <strong>Node Count:</strong> ${summary.node_count || summary.nodes?.length || 0}
        </div>
    </div>`;

    html += '<div class="detail-section"><h4>Inputs</h4>';
    if (summary.inputs && summary.inputs.length > 0) {
      html += '<ul class="model-io-list">';
      summary.inputs.forEach((input) => {
        const shape = input.shape ? `[${input.shape.join(", ")}]` : "";
        html += `<li class="model-io-item">
                <strong>${input.name || "Unnamed"}</strong>
                ${input.type ? `<span class="io-type">${input.type}</span>` : ""}
                ${shape ? `<span class="io-shape">${shape}</span>` : ""}
            </li>`;
      });
      html += "</ul>";
    } else {
      html += "<p>No input information available</p>";
    }
    html += "</div>";

    html += '<div class="detail-section"><h4>Outputs</h4>';
    if (summary.outputs && summary.outputs.length > 0) {
      html += '<ul class="model-io-list">';
      summary.outputs.forEach((output) => {
        const shape = output.shape ? `[${output.shape.join(", ")}]` : "";
        html += `<li class="model-io-item">
                <strong>${output.name || "Unnamed"}</strong>
                ${output.type ? `<span class="io-type">${output.type}</span>` : ""}
                ${shape ? `<span class="io-shape">${shape}</span>` : ""}
            </li>`;
      });
      html += "</ul>";
    } else {
      html += "<p>No output information available</p>";
    }
    html += "</div>";

    html += '<div class="detail-section"><h4>Layer Types</h4>';
    if (Object.keys(opTypeCounts).length > 0) {
      html += '<table class="layer-types-table">';
      html += "<thead><tr><th>Operation</th><th>Count</th></tr></thead>";
      html += "<tbody>";

      Object.entries(opTypeCounts).forEach(([opType, count]) => {
        html += `<tr>
                <td>${opType}</td>
                <td>${count} node${count > 1 ? "s" : ""}</td>
            </tr>`;
      });

      const totalNodes = summary.nodes.length;
      html += `<tr class="total-row">
            <td><strong>Total</strong></td>
            <td><strong>${totalNodes} node${totalNodes > 1 ? "s" : ""}</strong></td>
        </tr>`;

      html += "</tbody></table>";
    } else {
      html += "<p>No layer information available</p>";
    }
    html += "</div>";

    html += "</div>";

    modelDetailsElement.innerHTML = html;
  }

  async function handleCsvUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    try {
      event.target.value = "";

      updateStatus(`Loading ${file.name}...`);
      showSpinner(true);

      const arrayBuffer = await readFileAsArrayBuffer(file);

      const base64 = arrayBufferToBase64(arrayBuffer);

      const result = await window.eel.load_csv_data(base64, file.name)();

      if (result.error) {
        updateStatus(`Error: ${result.error}`);
        console.error("CSV loading error:", result.error);
        return;
      }

      console.log("CSV data loaded:", result);

      window.csvData = {
        data: result.data,
        summary: result.summary,
        filename: file.name,
        total: result.summary.rows,
      };

      updateStatus(
        `Dataset loaded: ${file.name} (${result.summary.rows} rows, ${result.summary.columns} columns)`,
      );

      displayCsvData(window.csvData);

      updateDataSummary(result.summary);

      const preprocessButton = document.getElementById("btn-preprocess");
      const clearDataButton = document.getElementById("btn-clear-data");

      if (preprocessButton) {
        preprocessButton.disabled = false;
        console.log("Preprocess button enabled");
      }

      if (clearDataButton) {
        clearDataButton.disabled = false;
        clearDataButton.classList.remove("disabled");
        clearDataButton.style.pointerEvents = "auto";
        console.log("Clear data button enabled");
      }
    } catch (error) {
      console.error("Error loading CSV:", error);
      updateStatus(`Error: ${error.message}`);
    } finally {
      showSpinner(false);
    }
    updateOptimizationUI();
  }

  async function loadCsvPage() {
    try {
      showSpinner(true);

      const pageData = await window.eel.get_csv_page(currentPage, pageSize)();

      if (pageData.error) {
        updateStatus(`Error: ${pageData.error}`);
        return;
      }

      updatePaginationInfo();

      renderCsvTable(pageData.data);
    } catch (error) {
      console.error("Error loading page:", error);
      updateStatus(`Error: ${error.message}`);
    } finally {
      showSpinner(false);
    }
  }

  function renderCsvTable(data) {
    const table = document.getElementById("csv-table");
    if (!table) return;

    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");

    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (!data || data.length === 0) {
      tbody.innerHTML = '<tr><td colspan="100">No data available</td></tr>';
      return;
    }

    const headerRow = document.createElement("tr");
    const columns = Object.keys(data[0]);

    const indexHeader = document.createElement("th");
    indexHeader.textContent = "#";
    headerRow.appendChild(indexHeader);

    columns.forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column;
      headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);

    data.forEach((row, index) => {
      const tr = document.createElement("tr");

      const indexCell = document.createElement("td");
      indexCell.textContent = currentPage * pageSize + index + 1;
      indexCell.className = "row-index";
      tr.appendChild(indexCell);

      columns.forEach((column) => {
        const td = document.createElement("td");
        td.textContent = row[column] !== null ? row[column] : "";
        td.dataset.column = column;
        td.dataset.row = index;
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
  }

  function updatePaginationInfo() {
    const pageInfo = document.getElementById("page-info");
    const prevButton = document.getElementById("prev-page");
    const nextButton = document.getElementById("next-page");

    if (pageInfo) {
      pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
    }

    if (prevButton) {
      prevButton.disabled = currentPage === 0;
    }

    if (nextButton) {
      nextButton.disabled = currentPage >= totalPages - 1;
    }
  }

  function renderCsvTable(data) {
    const table = document.getElementById("csv-table");
    if (!table) return;

    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");

    thead.innerHTML = "";
    tbody.innerHTML = "";

    if (!data || data.length === 0) {
      tbody.innerHTML = '<tr><td colspan="100">No data available</td></tr>';
      return;
    }

    const headerRow = document.createElement("tr");
    const columns = Object.keys(data[0]);

    const indexHeader = document.createElement("th");
    indexHeader.textContent = "#";
    headerRow.appendChild(indexHeader);

    columns.forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column;
      headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);

    data.forEach((row, index) => {
      const tr = document.createElement("tr");

      const indexCell = document.createElement("td");
      indexCell.textContent = currentPage * pageSize + index + 1;
      indexCell.className = "row-index";
      tr.appendChild(indexCell);

      columns.forEach((column) => {
        const td = document.createElement("td");
        td.textContent = row[column] !== null ? row[column] : "";
        td.dataset.column = column;
        td.dataset.row = index;
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
  }

  async function preprocessData() {
    try {
      showSpinner(true);

      const missingValues = document.getElementById("missing-values").value;
      const scalingMethod = document.getElementById("scaling-method").value;
      const shuffleData = document.getElementById("shuffle-data").checked;

      const result = await window.eel.preprocess_data(
        missingValues,
        scalingMethod,
        shuffleData,
      )();

      if (result.error) {
        updateStatus(`Error: ${result.error}`);
        return;
      }

      csvData.rowCount = result.rows;
      csvData.processed = true;

      currentPage = 0;
      totalPages = Math.ceil(result.rows / pageSize);

      updateStatus(`Data preprocessed: ${result.operations.join(", ")}`);

      loadCsvPage();
    } catch (error) {
      console.error("Error preprocessing data:", error);
      updateStatus(`Error: ${error.message}`);
    } finally {
      showSpinner(false);
    }
  }

  async function generateModelSummary() {
    try {
      showSpinner(true);

      const summary = await window.eel.generate_model_summary()();

      if (summary.error) {
        console.error("Error generating summary:", summary.error);
        return;
      }

      displayModelInfo(summary.model_info);

      displayLayersInfo(summary.layers);
    } catch (error) {
      console.error("Error generating model summary:", error);
    } finally {
      showSpinner(false);
    }
  }

  function displayModelInfo(modelInfo) {
    const modelInfoElement = document.getElementById("model-info");
    const modelDetailsElement = document.getElementById("model-details");

    if (!modelInfo) return;

    if (modelInfoElement) {
      modelInfoElement.innerHTML = `
                <table class="info-table">
                    <tr>
                        <th>Producer</th>
                        <td>${modelInfo.producer || "Unknown"}</td>
                    </tr>
                    <tr>
                        <th>Version</th>
                        <td>${modelInfo.producer_version || "Unknown"}</td>
                    </tr>
                    <tr>
                        <th>IR Version</th>
                        <td>${modelInfo.ir_version || "Unknown"}</td>
                    </tr>
                    <tr>
                        <th>Domain</th>
                        <td>${modelInfo.domain || "Unknown"}</td>
                    </tr>
                </table>
            `;
    }

    if (modelDetailsElement) {
      let inputShapesHtml = "";
      if (summary.inputs) {
        summary.inputs.forEach((input) => {
          inputShapesHtml += `
                        <div class="detail-item">
                            <strong>${input.name}:</strong> [${input.shape.join(", ")}]
                            <span class="detail-type">${input.data_type}</span>
                        </div>
                    `;
        });
      }

      let outputShapesHtml = "";
      if (summary.outputs) {
        summary.outputs.forEach((output) => {
          outputShapesHtml += `
                        <div class="detail-item">
                            <strong>${output.name}:</strong> [${output.shape.join(", ")}]
                            <span class="detail-type">${output.data_type}</span>
                        </div>
                    `;
        });
      }

      modelDetailsElement.innerHTML = `
                <div class="detail-section">
                    <h4>Inputs</h4>
                    ${inputShapesHtml || "<p>No input information available</p>"}
                </div>
                <div class="detail-section">
                    <h4>Outputs</h4>
                    ${outputShapesHtml || "<p>No output information available</p>"}
                </div>
                <div class="detail-section">
                    <h4>Model Information</h4>
                    <div class="detail-item">
                        <strong>Producer:</strong> ${modelInfo.producer || "Unknown"}
                    </div>
                    <div class="detail-item">
                        <strong>Version:</strong> ${modelInfo.producer_version || "Unknown"}
                    </div>
                </div>
            `;
    }
  }

  function displayLayersInfo(layers) {
    const layersTableBody = document.querySelector("#layers-table tbody");
    if (!layersTableBody) return;

    layersTableBody.innerHTML = "";

    if (!layers || layers.length === 0) {
      layersTableBody.innerHTML =
        '<tr><td colspan="3">No layer information available</td></tr>';
      return;
    }

    layers.forEach((layer, index) => {
      const tr = document.createElement("tr");

      const nameCell = document.createElement("td");
      nameCell.textContent = layer.name || `Layer ${index + 1}`;

      const typeCell = document.createElement("td");
      typeCell.textContent = layer.type || "Unknown";

      const detailsCell = document.createElement("td");

      let details = "";
      if (layer.type === "Conv" && layer.kernel_size) {
        details = `Kernel: ${layer.kernel_size.join("×")}`;
      } else if (layer.type === "Linear" && layer.shape) {
        details = `Shape: [${layer.shape.join(", ")}]`;
      }

      detailsCell.textContent = details;

      tr.appendChild(nameCell);
      tr.appendChild(typeCell);
      tr.appendChild(detailsCell);

      layersTableBody.appendChild(tr);
    });
  }

  function populateFeatureSelectors(columnsInfo) {
    const featureCheckboxes = document.getElementById("feature-checkboxes");
    const targetSelect = document.getElementById("target-select");

    if (!featureCheckboxes || !targetSelect) return;

    featureCheckboxes.innerHTML = "";
    targetSelect.innerHTML = '<option value="">-- Select Target --</option>';

    if (!columnsInfo || columnsInfo.length === 0) {
      featureCheckboxes.innerHTML = "<p>No columns available</p>";
      return;
    }

    columnsInfo.forEach((column) => {
      const label = document.createElement("label");
      label.className = "feature-checkbox";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = column.name;
      checkbox.name = "features";
      checkbox.dataset.type = column.dtype;

      checkbox.addEventListener("change", checkEnableTestButton);

      const span = document.createElement("span");
      span.textContent = column.name;

      label.appendChild(checkbox);
      label.appendChild(span);

      featureCheckboxes.appendChild(label);

      const option = document.createElement("option");
      option.value = column.name;
      option.textContent = column.name;
      targetSelect.appendChild(option);
    });

    targetSelect.disabled = false;

    targetSelect.addEventListener("change", checkEnableTestButton);
  }

  function checkEnableTestButton() {
    const runTestButton = document.getElementById("run-test");
    if (!runTestButton) return;

    const featureCheckboxes = document.querySelectorAll(
      'input[name="features"]:checked',
    );
    const targetSelect = document.getElementById("target-select");

    const hasFeatures = featureCheckboxes.length > 0;
    const hasTarget = targetSelect && targetSelect.value;
    const hasModelAndData = network !== null && csvData !== null;

    runTestButton.disabled = !(hasFeatures && hasTarget && hasModelAndData);
  }

  async function getArchitectureSuggestions() {
    try {
      showSpinner(true);
      updateStatus("Generating architecture suggestions...");

      document.getElementById("tab-optimize").click();

      const complexity = document.querySelector(
        'input[name="complexity"]:checked',
      ).value;

      const result = await window.eel.suggest_architecture(complexity)();

      if (result.error) {
        updateStatus(`Error: ${result.error}`);
        console.error(result.error);
        return;
      }

      updateStatus("Architecture suggestions generated");

      displayArchitectureSuggestions(result.architecture);

      document
        .getElementById("architecture-visualization")
        .classList.remove("hidden");

      createArchitectureVisualization(result.architecture);
    } catch (error) {
      console.error("Error getting suggestions:", error);
      updateStatus(`Error: ${error.message}`);
    } finally {
      showSpinner(false);
    }
  }

  function displayArchitectureSuggestions(architecture) {
    const container = document.getElementById("architecture-display");
    if (!container) return;

    const hiddenLayersStr = architecture.hidden_layers.join(", ");

    container.innerHTML = `
            <div class="architecture-card">
                <h3>Recommended Network Structure</h3>
                <div class="architecture-details">
                    <div class="layer-info">
                        <strong>Input Layer:</strong> ${architecture.input_layer} neurons
                    </div>
                    <div class="layer-info">
                        <strong>Hidden Layers:</strong> ${architecture.hidden_layers.length} layers [${hiddenLayersStr}]
                    </div>
                    <div class="layer-info">
                        <strong>Output Layer:</strong> ${architecture.output_layer} neurons
                    </div>
                </div>
            </div>
            
            <div class="architecture-card">
                <h3>Training Parameters</h3>
                <div class="architecture-details">
                    <div class="param-info">
                        <strong>Activation Function:</strong> ${architecture.activation}
                    </div>
                    <div class="param-info">
                        <strong>Learning Rate:</strong> ${architecture.learning_rate}
                    </div>
                    <div class="param-info">
                        <strong>Batch Size:</strong> ${architecture.batch_size}
                    </div>
                    <div class="param-info">
                        <strong>Epochs:</strong> ${architecture.epochs}
                    </div>
                </div>
            </div>
        `;
  }

  function createArchitectureVisualization(architecture) {
    const canvas = document.getElementById("architecture-canvas");
    if (!canvas) return;

    if (architectureChart) {
      architectureChart.destroy();
    }

    const layers = [
      architecture.input_layer,
      ...architecture.hidden_layers,
      architecture.output_layer,
    ];

    const layerLabels = [
      "Input",
      ...architecture.hidden_layers.map((_, i) => `Hidden ${i + 1}`),
      "Output",
    ];

    const ctx = canvas.getContext("2d");
    architectureChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: layerLabels,
        datasets: [
          {
            label: "Neurons",
            data: layers,
            backgroundColor: [
              "rgba(54, 162, 235, 0.7)",
              ...Array(architecture.hidden_layers.length).fill(
                "rgba(75, 192, 192, 0.7)",
              ),
              "rgba(255, 99, 132, 0.7)",
            ],
            borderColor: [
              "rgba(54, 162, 235, 1)",
              ...Array(architecture.hidden_layers.length).fill(
                "rgba(75, 192, 192, 1)",
              ),
              "rgba(255, 99, 132, 1)",
            ],
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: "Neural Network Architecture",
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `${context.raw} neurons`;
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: "Number of Neurons",
            },
          },
        },
      },
    });
  }

  function showSpinner(show) {
    const spinner = document.querySelector(".spinner");
    if (spinner) {
      spinner.style.display = show ? "block" : "none";
    }

    document.body.style.cursor = show ? "wait" : "default";
  }

  function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = (e) => reject(new Error("File reading failed"));
      reader.readAsArrayBuffer(file);
    });
  }

  function arrayBufferToBase64(buffer) {
    const uint8Array = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < uint8Array.byteLength; i++) {
      binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
  }

  setTimeout(checkTabVisibility, 1000);
});

window.addEventListener("resize", function () {
  if (network) {
    setTimeout(() => {
      network.redraw();
      network.fit();
    }, 200);
  }
});

function displayModelInfo(summary) {
  const detailsElement = document.getElementById("model-details");
  if (!summary || !detailsElement) return;

  let html = `
        <div class="detail-section">
            <h4>Basic Info</h4>
            <p><strong>Producer:</strong> ${summary.producer || "Unknown"}</p>
            <p><strong>IR Version:</strong> ${summary.ir_version || "Unknown"}</p>
            <p><strong>Domain:</strong> ${summary.domain || "General"}</p>
        </div>

        <div class="detail-section">
            <h4>Inputs (${summary.inputs?.length || 0})</h4>
            <ul>
                ${(summary.inputs || []).map((i) => `<li>${i.name} - ${i.type}</li>`).join("")}
            </ul>
        </div>

        <div class="detail-section">
            <h4>Outputs (${summary.outputs?.length || 0})</h4>
            <ul>
                ${(summary.outputs || []).map((o) => `<li>${o.name} - ${o.type}</li>`).join("")}
            </ul>
        </div>
    `;

  detailsElement.innerHTML = html;
}

function populateFeatureSelectors(columns) {
  const featureContainer = document.getElementById("feature-checkboxes");
  const targetSelect = document.getElementById("target-select");

  featureContainer.innerHTML = "";
  targetSelect.innerHTML = '<option value="">-- Select Target --</option>';

  columns.forEach((column) => {
    const label = document.createElement("label");
    label.className = "feature-label";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = column;
    checkbox.name = "features";

    const span = document.createElement("span");
    span.textContent = column;

    label.appendChild(checkbox);
    label.appendChild(span);
    featureContainer.appendChild(label);

    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    targetSelect.appendChild(option);
  });

  targetSelect.disabled = false;
}

function displayArchitectureSuggestions(result) {
  const container = document.getElementById("architecture-display");
  if (!container) return;

  container.innerHTML = `
        <div class="architecture-card">
            <h3>Recommended Architecture</h3>
            <p>${result.recommendation}</p>
            <div class="details">
                <p><strong>Hidden Layers:</strong> ${result.layers}</p>
                <p><strong>Neurons per Layer:</strong> ${result.neurons}</p>
                <p><strong>Learning Rate:</strong> ${result.learning_rate}</p>
                <p><strong>Batch Size:</strong> ${result.batch_size}</p>
            </div>
        </div>
    `;
}

async function applyPostprocessing() {
  const processingSteps = {
    inverse_scaling: document.getElementById("inverse-scaling").checked,
    decode_predictions: document.getElementById("decode-predictions").checked,
  };

  try {
    const result = await window.eel.postprocess_data(processingSteps)();

    if (result.error) {
      showError(result.error);
      return;
    }

    updateDataPreview(result.data);
    showSuccess(result.message);
  } catch (error) {
    showError(`Postprocessing failed: ${error.message}`);
  }
}

document.getElementById("debugBtn").addEventListener("click", function () {
  const tabButton = document.getElementById("tab-optimize");
  if (tabButton) {
    tabButton.click();

    setTimeout(() => {
      forceCheckOptimizationTab();

      document.getElementById("content-optimize").style =
        "display: block !important;";

      console.log("Final visibility check completed");
    }, 500);
  }
});
