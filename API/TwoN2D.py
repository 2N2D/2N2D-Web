import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from FileHandler import (createTempFile)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_debug.log"),
        logging.StreamHandler()
    ]
)

def load_onnx_model(file_bytes):
    try:
        temp_file_path = createTempFile(file_bytes, ".onnx")

        model = onnx.load(temp_file_path)
        graph = model.graph


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

        os.remove(temp_file_path)

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary,
        }
    
    except Exception as e:
        logging.exception("Failed to load ONNX model")
        return {"error": str(e)}


def load_csv_data(binary_data, filename):
    try:

        data_io = io.BytesIO(binary_data)
        df = pd.read_csv(data_io)
        
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
    
def analyze_onnx_model(model_bytes):
    try:
        tempFilePath = createTempFile(model_bytes, ".onnx")
        model = onnx.load(tempFilePath)
        
        graph = model.graph
        input_shapes = {}
        output_shapes = {}
        
        for input in graph.input:
            shape = []
            if hasattr(input.type.tensor_type, 'shape'):
                for dim in input.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(None)
                    else:
                        shape.append(dim.dim_value)
            input_shapes[input.name] = shape
        
        for output in graph.output:
            shape = []
            if hasattr(output.type.tensor_type, 'shape'):
                for dim in output.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(None)
                    else:
                        shape.append(dim.dim_value)
            output_shapes[output.name] = shape
        op_counts = {}
        for node in graph.node:
            if node.op_type not in op_counts:
                op_counts[node.op_type] = 0
            op_counts[node.op_type] += 1
        linear_layers = 0
        for node in graph.node:
            if node.op_type in ['Gemm', 'MatMul', 'Conv']:
                linear_layers += 1
        activation_layers = 0
        for node in graph.node:
            if node.op_type in ['Relu', 'Sigmoid', 'Tanh', 'LeakyRelu']:
                activation_layers += 1
        lstm_layers = 0
        for node in graph.node:
            if node.op_type == 'LSTM':
                lstm_layers += 1
        input_features = 0
        for input_name, shape in input_shapes.items():
            if len(shape) >= 2:  # Assuming batch dimension is first
                input_features = shape[1] if shape[1] is not None else 0
        
        output_features = 0
        for output_name, shape in output_shapes.items():
            if len(shape) >= 2:  # Assuming batch dimension is first
                output_features = shape[1] if shape[1] is not None else 0
        hidden_dims = []
        for node in graph.node:
            if node.op_type == 'Gemm':
                for attr in node.attribute:
                    if attr.name == 'transB' and attr.i == 1:  # Common pattern in ONNX
                        for init in graph.initializer:
                            if init.name == node.input[1]:  # Weight tensor
                                hidden_dims.append(init.dims[0])
        if not hidden_dims and input_features > 0 and output_features > 0:
            hidden_dims = [max(input_features, output_features) * 2]
        is_recurrent = lstm_layers > 0
        if is_recurrent:
            num_layers = max(1, lstm_layers)
            model_type = 'lstm'
        else:
            num_layers = max(1, min(linear_layers, activation_layers))
            model_type = 'feedforward'
        if hidden_dims:
            hidden_size = max(hidden_dims)
        else:
            hidden_size = max(64, input_features * 2)  # Reasonable default
        
        os.remove(tempFilePath)

        return {
            "model_type": model_type,
            "input_size": input_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_features if output_features > 0 else 1,
            "is_recurrent": is_recurrent,
            "layers": num_layers,  # For backward compatibility
            "neurons": hidden_size  # For backward compatibility
        }
    except Exception as e:
        logging.exception("Error analyzing ONNX model")
        os.remove(tempFilePath)
        return {"error": str(e), "layers": 1, "neurons": 64}


class StructurePreservingModel(nn.Module):
    """
    PyTorch model that preserves the structure of the original ONNX model.
    Supports both feedforward and LSTM architectures with input/output adapters.
    """
    def __init__(self, model_type, input_size, hidden_size, num_layers, output_size, 
                 actual_input_size=None, actual_output_size=None):
        super(StructurePreservingModel, self).__init__()
        self.needs_input_adapter = actual_input_size is not None and actual_input_size != input_size
        
        if self.needs_input_adapter:
            self.input_adapter = nn.Linear(actual_input_size, input_size)
            self.input_size = actual_input_size
        else:
            self.input_size = input_size
        
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))
            
            self.model = nn.Sequential(*layers)
        self.needs_output_adapter = actual_output_size is not None and actual_output_size != output_size
        
        if self.needs_output_adapter:
            self.output_adapter = nn.Linear(output_size, actual_output_size)
    
    def forward(self, x):
        if self.needs_input_adapter:
            x = self.input_adapter(x)
        if self.model_type == 'lstm':
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            lstm_out, _ = self.lstm(x)
            x = self.fc(lstm_out[:, -1, :])
        else:
            x = self.model(x)
        if self.needs_output_adapter:
            x = self.output_adapter(x)
        
        return x


def find_optimal_architecture(onnx_bytes, csv_bytes, input_features, target_feature, max_epochs=10, status_callback=None):
    
    def send_status(message):
        if status_callback:
            status_callback(message)
        else:
            logging.info(message.get("status", str(message)))

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(4)  # Limit CPU threads for better stability
        
        if csv_bytes is None:
            send_status({"status": "No data loaded", "error": True})
            return {"error": "No data loaded"}

        if onnx_bytes is None:
            send_status({"status": "No ONNX model loaded", "error": True})
            return {"error": "No ONNX model loaded"}

        send_status({"status": "Analyzing model...", "progress": 5})
        model_info = analyze_onnx_model(onnx_bytes)
        base_layers = model_info.get('layers', 1)
        base_neurons = model_info.get('neurons', 64)
        model_type = model_info.get('model_type', 'feedforward')
        input_size = model_info.get('input_size', 0)
        output_size = model_info.get('output_size', 1)
        # is_recurrent = model_info.get('is_recurrent', False)

        send_status({
            "status": f"Base model: {model_type}, {base_layers} layers, {base_neurons} neurons",
            "progress": 15
        })

        data_io = io.BytesIO(csv_bytes)
        df = pd.read_csv(data_io)

        missing_values = df[input_features + [target_feature]].isna().sum()
        if missing_values.sum() > 0:
            logging.warning(f"Found missing values in data: {missing_values[missing_values > 0]}")
            logging.info("Filling missing values with column means")
            df[input_features + [target_feature]] = df[input_features + [target_feature]].fillna(df[input_features + [target_feature]].mean())
        
        X = df[input_features].values.astype(np.float32)
        y = df[target_feature].values.astype(np.float32).reshape(-1, 1)
        if np.isnan(X).any() or np.isinf(X).any():
            logging.warning("Found NaN or infinite values in features. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(y).any() or np.isinf(y).any():
            logging.warning("Found NaN or infinite values in target. Replacing with zeros.")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device="cpu")
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device="cpu")
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device="cpu")
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device="cpu")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        actual_input_size = X.shape[1]
        actual_output_size = y.shape[1]
        
        logging.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        logging.info(f"Feature shape: {X_train.shape}, Target shape: {y_train.shape}")
        logging.info(f"Actual input size: {actual_input_size}, Actual output size: {actual_output_size}")
        logging.info(f"Model expected input size: {input_size}, output size: {output_size}")
        neurons_options = [max(16, base_neurons // 2), base_neurons, base_neurons * 2]
        layers_options = [max(1, base_layers - 1), base_layers, base_layers + 1]

        results = []
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
                model = StructurePreservingModel(
                    model_type=model_type,
                    input_size=input_size if input_size > 0 else actual_input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size if output_size > 0 else 1,
                    actual_input_size=actual_input_size,
                    actual_output_size=actual_output_size
                )
                model = model.to("cpu")
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                for epoch in range(max_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to("cpu"), batch_y.to("cpu")
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        if outputs.shape != batch_y.shape:
                            if outputs.shape[0] == batch_y.shape[0]:  # Same batch size
                                if outputs.shape[1] > batch_y.shape[1]:
                                    outputs = outputs[:, :batch_y.shape[1]]
                                elif outputs.shape[1] < batch_y.shape[1]:
                                    padding = torch.zeros(outputs.shape[0], batch_y.shape[1] - outputs.shape[1], device="cpu")
                                    outputs = torch.cat([outputs, padding], dim=1)
                        
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    all_outputs = []
                    all_targets = []
                    
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to("cpu"), batch_y.to("cpu")
                        outputs = model(batch_X)
                        if outputs.shape != batch_y.shape:
                            if outputs.shape[0] == batch_y.shape[0]:  # Same batch size
                                if outputs.shape[1] > batch_y.shape[1]:
                                    outputs = outputs[:, :batch_y.shape[1]]
                                elif outputs.shape[1] < batch_y.shape[1]:
                                    padding = torch.zeros(outputs.shape[0], batch_y.shape[1] - outputs.shape[1], device="cpu")
                                    outputs = torch.cat([outputs, padding], dim=1)
                        
                        all_outputs.append(outputs.cpu().numpy())
                        all_targets.append(batch_y.cpu().numpy())
                    
                    all_outputs = np.vstack(all_outputs)
                    all_targets = np.vstack(all_targets)
                    if all_outputs.shape[1] != all_targets.shape[1]:
                        if all_outputs.shape[1] > all_targets.shape[1]:
                            all_outputs = all_outputs[:, :all_targets.shape[1]]
                        else:
                            all_targets = all_targets[:, :all_outputs.shape[1]]
                    
                    test_loss = mean_squared_error(all_targets, all_outputs)
                    r2 = r2_score(all_targets, all_outputs)

                results.append({
                    "neurons": hidden_size, 
                    "layers": num_layers, 
                    "test_loss": test_loss, 
                    "r2_score": r2
                })
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_result = results[-1]
                    best_model = model
                    send_status({
                        "status": f"New best config: {num_layers} layers, {hidden_size} neurons",
                        "progress": progress_percent
                    })

        send_status({"status": "Final training of best model...", "progress": 90})
        best_model = best_model.to("cpu")
        dummy_input = torch.randn(1, actual_input_size, device="cpu")
        onnx_path = os.path.join(tempfile.gettempdir(), "optimized_model.onnx")
        torch.onnx.export(
            best_model, 
            dummy_input, 
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12,  # Use a higher opset version for better structure preservation
            do_constant_folding=False,  # Disable constant folding to preserve structure
            export_params=True,
            keep_initializers_as_inputs=True,  # Keep initializers as inputs to preserve structure
            verbose=True
        )
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, onnx_path)

        send_status({"status": "Optimization complete", "progress": 100})
        return {
            "results": results,
            "best_config": best_result,
            "model_path": onnx_path,
        }
    except Exception as e:
        logging.exception("Error in find_optimal_architecture")
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