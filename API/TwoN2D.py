# Standard library
import os
import io
import base64
import tempfile
import logging

# Third-party libraries
import numpy as np
import pandas as pd
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

##ROBI, NU AM TESTAT INCA NIMIC
# Nu am avut cum sa testez inca, dar teoretic ar trebui sa mearga, deoarece din logica nu am sters
# doar ce a tinut de eel, eu lucrez in principal pe engleza si folosesc camelCase pentru variabile, ca sa stii si tu
# Read me-ul e facut cu chatgpt ca sa fie acolo, ca mi-e lene momentan sa il fac, deci daca sunt balari pe acolo, ayaye

# Architecture:
# -API
# 2n2d.py - api in sine, nu are nimic legat de web, functioneaza ca o librarie
# 2n2dEndPoint.py - cum zice in nume, este un endpoint pentru api, care este conectat la web si chestii, asa putem adauga mai tarziu mai usor alte chestii
#                   pe langa asta, daca adaugam ceva protectie si chestii, va fi mai simplu asa
# -WEB INTERFACE
# -OLD
# proiectul vechi pe care il tin pentru referinta si sa copiez codul necesar

# TO DO
# Make functions client based and general, not using global variables
# More refactoring + comments
# Next js front end remake
# refact js middleware
# Read me pe git

current_model = None
current_model_path = None
current_data = None

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_debug.log"),
        logging.StreamHandler()
    ]
)


# def test_connection():
#     return {"status": "success", "message": "Python backend is connected"}

def load_onnx_model(base64_str):
    global current_model, current_model_path
    try:
        file_bytes = base64.b64decode(base64_str)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        model = onnx.load(temp_file_path)
        graph = model.graph

        current_model = model
        current_model_path = temp_file_path

        nodes = []
        edges = []
        node_map = {}

        for i, node in enumerate(graph.node):
            label = node.name if node.name else node.op_type
            nodes.append({
                "id": i,
                "label": label,
                "title": str(node),
                "group": "operation",
            })
            if node.output:
                node_map[node.output[0]] = i

        for i, node in enumerate(graph.node):
            for input_name in node.input:
                if input_name in node_map:
                    edges.append({"from": node_map[input_name], "to": i})

        node_summary = []
        for node in graph.node:
            node_summary.append({
                "name": node.name if node.name else "Unnamed",
                "op_type": node.op_type
            })

        summary = {
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "inputs": [{"name": inp.name} for inp in graph.input],
            "outputs": [{"name": out.name} for out in graph.output],
            "nodes": node_summary,
            "node_count": len(graph.node)
        }

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary
        }
    except Exception as e:
        logging.exception("Failed to load ONNX model")
        return {"error": str(e)}


def load_csv_data(base64_data, filename):
    global current_data
    try:
        binary_data = base64.b64decode(base64_data)
        data_io = io.BytesIO(binary_data)
        df = pd.read_csv(data_io)
        current_data = df
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "filename": filename,
            "missing_values": df.isna().sum().to_dict(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        return {"data": df.head(50).to_dict('records'), "summary": summary}
    except Exception as e:
        logging.exception("Failed to load CSV data")
        return {"error": str(e)}


def analyze_onnx_model(model):
    try:
        graph = model.graph
        linear_ops = [node for node in graph.node if node.op_type in ['Gemm', 'MatMul']]
        num_layers = len(linear_ops) - 1
        neurons = []
        for node in linear_ops:
            for input_name in node.input[1:]:
                for init in graph.initializer:
                    if init.name == input_name:
                        dims = list(init.dims)
                        if len(dims) >= 2:
                            neurons.append(dims[0])
        avg_neurons = int(np.median(neurons)) if neurons else 64
        return {"layers": max(1, num_layers), "neurons": avg_neurons}
    except Exception as e:
        logging.exception("Error analyzing ONNX model")
        return {"error": str(e)}


def find_optimal_architecture(input_features, target_feature, max_epochs=10, status_callback=None):
    def send_status(message):
        if status_callback:
            status_callback(message)
        else:
            logging.info(message.get("status", str(message)))

    global current_data, current_model
    try:
        if current_data is None:
            send_status({"status": "No data loaded", "error": True})
            return {"error": "No data loaded"}

        if current_model is None:
            send_status({"status": "No ONNX model loaded", "error": True})
            return {"error": "No ONNX model loaded"}

        send_status({"status": "Analyzing model...", "progress": 5})

        model_info = analyze_onnx_model(current_model)
        base_layers = model_info.get('layers', 1)
        base_neurons = model_info.get('neurons', 64)

        send_status({"status": f"Base model: {base_layers} layers, {base_neurons} neurons", "progress": 10})

        X = current_data[input_features].values.astype(np.float32)
        y = current_data[target_feature].values.astype(np.float32).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        neurons_options = [max(16, base_neurons // 2), base_neurons, base_neurons * 2]
        layers_options = [max(1, base_layers - 1), base_layers, base_layers + 1]

        results = []

        def create_model(input_size, hidden_size, num_layers):
            layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            layers.append(nn.Linear(hidden_size, 1))
            return nn.Sequential(*layers)

        best_loss = float('inf')
        current_iteration = 0
        total_iterations = len(neurons_options) * len(layers_options)

        for hidden_size in neurons_options:
            for num_layers in layers_options:
                current_iteration += 1
                progress_percent = 20 + int((current_iteration / total_iterations) * 60)

                send_status({
                    "status": f"Testing {num_layers} layers, {hidden_size} neurons",
                    "progress": progress_percent
                })

                model = create_model(X_train.shape[1], hidden_size, num_layers)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(max_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    test_loss = criterion(model(torch.FloatTensor(X_test)), torch.FloatTensor(y_test)).item()
                    r2 = r2_score(y_test, model(torch.FloatTensor(X_test)).numpy())

                results.append({"neurons": hidden_size, "layers": num_layers, "test_loss": test_loss, "r2_score": r2})
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_result = results[-1]
                    send_status({"status": f"New best config: {num_layers} layers, {hidden_size} neurons",
                                 "progress": progress_percent})

        send_status({"status": "Final training of best model...", "progress": 90})
        best_model = create_model(X_train.shape[1], best_result["neurons"], best_result["layers"])
        best_model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(best_model.parameters(), lr=0.001)

        for epoch in range(max_epochs):
            for batch_X, batch_y in train_loader:
                outputs = best_model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dummy_input = torch.randn(1, X_train.shape[1])
        onnx_path = os.path.join(tempfile.gettempdir(), "optimized_model.onnx")
        torch.onnx.export(best_model, dummy_input, onnx_path, input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

        send_status({"status": "Optimization complete", "progress": 100})
        return {
            "results": results,
            "best_config": best_result,
            "model_path": onnx_path
        }
    except Exception as e:
        send_status({"status": f"Error: {str(e)}", "error": True, "progress": 0})
        return {"error": str(e)}


def download_optimized_model(path):
    try:
        with open(path, "rb") as f:
            model_bytes = f.read()
        return {
            "base64": base64.b64encode(model_bytes).decode("utf-8"),
            "filename": "optimized_model.onnx"
        }
    except Exception as e:
        logging.exception("Failed to read optimized model")
        return {"error": str(e)}
