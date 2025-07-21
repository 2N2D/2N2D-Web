"use client";
// import { Chart } from "chart.js";
import {Network} from "vis-network";
import {DataSet} from "vis-data";
import {getFile} from "@/lib/fileHandler/r2Bucket"

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


export async function downloadFileRequest(path, filename, bucket = "2n2d") {
    try {
        const blob = await downloadFile(path);

        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

    } catch (error) {
        console.error("Download failed", error);
    }
}