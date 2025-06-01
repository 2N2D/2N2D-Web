"use client";
// import { Chart } from "chart.js";
import {Network} from "vis-network";
import {DataSet} from "vis-data";

// export async function createArchitectureComparisonChart(results, ctx) {
//   const dataByLayers = {};
//
//   results.forEach((result) => {
//     const layers = result.layers;
//     if (!dataByLayers[layers]) {
//       dataByLayers[layers] = [];
//     }
//     dataByLayers[layers].push({
//       neurons: result.neurons,
//       loss: result.test_loss,
//       r2: result.r2_score,
//     });
//   });
//
//   const datasets = [];
//   const colors = [
//     "rgba(54, 162, 235, 0.7)",
//     "rgba(255, 99, 132, 0.7)",
//     "rgba(75, 192, 192, 0.7)",
//   ];
//
//   Object.keys(dataByLayers).forEach((layers, index) => {
//     datasets.push({
//       label: `${layers} Layer${layers > 1 ? "s" : ""}`,
//       data: dataByLayers[layers].map((item) => ({
//         x: item.neurons,
//         y: item.loss,
//       })),
//       backgroundColor: colors[index % colors.length],
//       borderColor: colors[index % colors.length].replace("0.7", "1"),
//       borderWidth: 1,
//     });
//   });
//
//   new Chart(ctx, {
//     type: "scatter",
//     data: {
//       datasets: datasets,
//     },
//     options: {
//       responsive: true,
//       maintainAspectRatio: false,
//       plugins: {
//         tooltip: {
//           callbacks: {
//             label: function (context) {
//               const layerKey = context.dataset.label.split(" ")[0];
//               const item = dataByLayers[layerKey][context.dataIndex];
//               return `${context.dataset.label}, ${item.neurons} neurons: MSE = ${item.loss.toFixed(6)}, R² = ${item.r2.toFixed(4)}`;
//             },
//           },
//         },
//         title: {
//           display: true,
//           text: "Architecture Performance Comparison",
//         },
//       },
//       scales: {
//         x: {
//           type: "linear",
//           position: "bottom",
//           title: {
//             display: true,
//             text: "Neurons Per Layer",
//           },
//         },
//         y: {
//           title: {
//             display: true,
//             text: "Test Loss (MSE)",
//           },
//         },
//       },
//     },
//   });
// }

const NODE_TYPE_CATEGORIES = {
    coreLayers: "Dense Conv1D Conv2D Conv3D LSTM GRU RNN BatchNormalization Dropout Flatten Embedding",
    elementwiseMath: "Add Sub Mul Div Pow Sqrt Abs Neg Exp Log Clip Round Floor Ceil",
    activations: "ReLU LeakyReLU Sigmoid Tanh Softmax GELU ELU SELU HardSigmoid HardSwish",
    reductions: "ReduceSum ReduceMean ReduceMax ReduceMin ArgMax ArgMin GlobalAveragePool GlobalMaxPool",
    tensorManip: "Reshape Transpose Permute ExpandDims Squeeze Concat Split Slice Gather Scatter Tile Repeat Pad Stack Unstack",
    controlInfra: "Identity NoOp Switch Merge LoopCond Cast Constant Placeholder Input StopGradient CheckNumerics",
    linearAlgebra: "MatMul BatchMatMul Gemm Inverse Det Eig QR",
    attentionOps: "Attention Query Key Value Softmax MultiHeadAttention ScaledDotProductAttention",
    normalizationDropout: "LayerNorm Dropout RMSNorm",
    inputOutput: "Input Placeholder Output"
};

export const categoryColorMap = {
    coreLayers: {normal: "#4A90E2", highlight: "#317AE0"},
    elementwiseMath: {normal: "#A67C52", highlight: "#B78A5B"},
    activations: {normal: "#7ED6DF", highlight: "#54B9C7"},
    reductions: {normal: "#E1B12C", highlight: "#F0C542"},
    tensorManip: {normal: "#9B59B6", highlight: "#B478D7"},
    controlInfra: {normal: "#95A5A6", highlight: "#BDC3C7"},
    linearAlgebra: {normal: "#27AE60", highlight: "#2ECC71"},
    attentionOps: {normal: "#F39C12", highlight: "#F5B041"},
    normalizationDropout: {normal: "#FF6B81", highlight: "#FF8A9B"},
    inputOutput: {normal: "#2C3E50", highlight: "#34495E"},
    other: {normal: "#777", highlight: "#999"} // fallback
};

export let nodes;
export let edges;

export function getNodeCategory(label) {
    for (const [category, keywords] of Object.entries(NODE_TYPE_CATEGORIES)) {
        if (keywords.split(" ").some(word => label.includes(word))) {
            return category;
        }
    }
    return "other";
}

function simplifyNodeName(node) {
    //Remove wierd stuff

    node.label = node.label.replace(/lstm|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/\/{2,}/g, '');
    node.label = node.label.replace(/model|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/statefeedback|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/attention|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/norm|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/featureprocessor|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/inputadapter|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/layer|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/output|[^a-zA-Z]/g, '');
    node.label = node.label.replace(/_/g, '')


    return node;
}

function colorNode(node) {
    const category = getNodeCategory(node.label);
    node.color = {
        border: categoryColorMap[category].normal,
        background: categoryColorMap[category].normal,
        highlight: {
            border: categoryColorMap[category].highlight,
            background: categoryColorMap[category].highlight,
        },
    };

    return node;
}

function assignLevels(nodes, edges) {
    const levelMap = {};
    const visited = new Set();

    const incomingMap = {};
    edges.forEach(e => {
        incomingMap[e.to] = (incomingMap[e.to] || 0) + 1;
    });

    const queue = nodes.filter(n => !incomingMap[n.id]);
    queue.forEach(n => levelMap[n.id] = 0);

    while (queue.length) {
        const node = queue.shift();
        visited.add(node.id);
        const level = levelMap[node.id];

        edges
            .filter(e => e.from === node.id)
            .forEach(e => {
                const nextId = e.to;
                if (!visited.has(nextId)) {
                    levelMap[nextId] = Math.max(levelMap[nextId] || 0, level + 1);
                    queue.push(nodes.find(n => n.id === nextId));
                }
            });
    }

    nodes.forEach(n => {
        n.level = levelMap[n.id] || 0;
    });

    return nodes;
}

export async function createVisualNetwork2D(results, ctx, constants, physicsEnabled, vertical, handleSelect) {
    let rawNodes = results.nodes;
    let rawEdges = results.edges;
    // rawNodes = assignLevels(rawNodes, rawEdges);
    nodes = new DataSet(rawNodes);
    edges = new DataSet(rawEdges);

    if (!constants) {

        let newEdges = [];
        let nodeMap = new Map(); // id → node
        let incomingEdges = new Map(); // id → array of incoming edges
        let outgoingEdges = new Map(); // id → array of outgoing edges

        rawNodes.forEach(node => nodeMap.set(node.id, node));
        rawEdges.forEach(edge => {
            if (!incomingEdges.has(edge.to)) incomingEdges.set(edge.to, []);
            if (!outgoingEdges.has(edge.from)) outgoingEdges.set(edge.from, []);
            incomingEdges.get(edge.to).push(edge);
            outgoingEdges.get(edge.from).push(edge);
        });

        let cleanNodes = [];
        let constantsToRemove = new Set();

        rawNodes.forEach(node => {
            if (node.label.includes("Constant")) {
                constantsToRemove.add(node.id);
            } else {
                cleanNodes.push(node);
            }
        });

        rawEdges.forEach(edge => {
            const from = edge.from;
            const to = edge.to;

            if (constantsToRemove.has(from)) {
                // This edge is from a Constant node
                let sources = incomingEdges.get(from) || [];
                sources.forEach(sourceEdge => {
                    newEdges.push({
                        from: sourceEdge.from,
                        to: to
                    });
                });
            } else if (constantsToRemove.has(to)) {
                // This edge goes to a Constant node
                let targets = outgoingEdges.get(to) || [];
                targets.forEach(targetEdge => {
                    newEdges.push({
                        from: from,
                        to: targetEdge.to
                    });
                });
            } else {
                // Normal edge
                newEdges.push(edge);
            }
        });

        const seen = new Set();
        newEdges = newEdges.filter(edge => {
            const key = `${edge.from}->${edge.to}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });


        nodes = new DataSet(cleanNodes);
        edges = new DataSet(newEdges);
    }


    nodes.forEach((node) => {
        node.title = node.title.replace(/doc_string:\s+"(?:[^"\\]|\\.|\\\n)*"/gs, '');
        node = simplifyNodeName(node);
        node = colorNode(node);
        node.font = {
            color: "#ffffff",
            size: 12,
            fontFamily: "Montserrat",
            align: "center",
        };
        node.group = "";
    });


    const options = {
        layout: {
            hierarchical: {
                enabled: true,
                direction: vertical ? "UD" : "LR",
                sortMethod: "hubsize",
                levelSeparation: 100,
                nodeSpacing: 60,
                treeSpacing: 60,
                blockShifting: true,
                edgeMinimization: true,
                parentCentralization: true,
            },
        },
        nodes: {
            shape: "box",
            margin: 10,
            font: {size: 14, face: "arial"},
            borderWidth: 1,
            shadow: true,
            widthConstraint: {maximum: 100}
        },
        edges: {
            arrows: {to: {enabled: true, scaleFactor: 0.5}},
            smooth: {
                enabled: true,
                type: "cubicBezier",
                forceDirection: "horizontal",
                roundness: 0.5
            }
        },
        physics: {
            enabled: physicsEnabled,
            stabilization: {
                enabled: true,
                iterations: 1000,
            },
            hierarchicalRepulsion: {
                centralGravity: 0.0,
                springLength: 40,
                springConstant: 0.01,
                nodeDistance: 40,
            },
            solver: "hierarchicalRepulsion",
        },
    };

    const network = new Network(ctx, {nodes, edges}, options);

    network.once("stabilizationIterationsDone", () => {
        network.fit({animation: {duration: 500}});
    });

    network.on("selectNode", (params) => {
        const nodeId = params.nodes[0];
        const node = nodes.get(nodeId);

        handleSelect(node);
    })

    return network;
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
        sphere.userData = {id: node.id, label: node.label, type: node.type};

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
