import eel
import onnx
import tempfile
import base64
import numpy as np
import pandas as pd
import os
import json
import io
import traceback
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import pyautogui  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
import sys


eel.init("web")


current_model = None
current_model_path = None
current_data = None
current_scaler = None

def debug_optimization(message):
    
    print(f"OPTIMIZATION DEBUG: {message}")
    try:
        
        eel.showDebugMessage(str(message))()
    except:
        pass
    


def debug_to_ui(message):
    try:
        eel.showDebugMessage(str(message))()
    except:
        logging.error("Failed to send debug message to UI")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.debug("Application starting")





@eel.expose
def test_connection():
    
    print("Eel connection test successful")
    return {"status": "success", "message": "Python backend is connected"}

@eel.expose
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
                "group": "operation"
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
        traceback.print_exc()
        return {"error": str(e)}

@eel.expose
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
        
        
        return {
            "data": df.head(50).to_dict('records'),
            "summary": summary
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@eel.expose
def get_csv_page(page=0, page_size=10):
    
    global current_data
    
    if current_data is None:
        return {"error": "No data loaded"}
    
    try:
        
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(current_data))
        
        
        page_data = current_data.iloc[start_idx:end_idx].to_dict('records')
        
        return {"data": page_data}
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@eel.expose
def generate_model_summary():
    
    global current_model
    
    if current_model is None:
        return {"error": "No model loaded"}
    
    try:
        model = current_model
        graph = model.graph
        
        
        inputs = [{"name": inp.name, "type": str(inp.type)} for inp in graph.input]
        outputs = [{"name": out.name, "type": str(out.type)} for out in graph.output]
        nodes = [{"name": node.name, "op_type": node.op_type} for node in graph.node]
        
        return {
            "inputs": inputs,
            "outputs": outputs,
            "nodes": nodes,
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "domain": model.domain
        }
        
    except Exception as e:
        return {"error": str(e)}
    
    

@eel.expose
def find_optimal_architecture(input_features, target_feature, max_epochs=10):
    
    global current_data, current_model
    
    debug_to_ui(f"Starting optimization with {len(input_features)} input features")
    
    logging.debug(f"==== OPTIMIZATION DEBUG ====")
    logging.debug(f"Called with input_features: {input_features}")
    logging.debug(f"Target feature: {target_feature}")
    logging.debug(f"Max epochs: {max_epochs}")
    logging.debug(f"Data available: {current_data is not None}")
    logging.debug(f"ONNX model loaded: {current_model is not None}")
    
    
    eel.updateOptimizationProgress({"status": "Starting optimization process...", "progress": 1})
    
    try:
        
        if current_data is None:
            print("ERROR: No data loaded")
            eel.updateOptimizationProgress({"status": "Error: No data loaded", "progress": 0, "error": True})
            return {"error": "No data loaded"}
            
        if current_model is None:
            print("ERROR: No ONNX model loaded")
            return {"error": "Please load an ONNX model first"}
        
        
        if not isinstance(input_features, list):
            print(f"WARNING: input_features is not a list but {type(input_features)}")
            if isinstance(input_features, str):
                input_features = [input_features]
                print(f"Converted to list: {input_features}")
                eel.updateOptimizationProgress({"status": "Converting input features to list...", "progress": 3})
        
        
        missing_cols = [col for col in input_features if col not in current_data.columns]
        if missing_cols:
            print(f"ERROR: Missing columns in dataframe: {missing_cols}")
            eel.updateOptimizationProgress({
                "status": f"Error: Columns not found in dataset: {', '.join(missing_cols)}", 
                "progress": 0, 
                "error": True
            })
            return {"error": f"Columns not found in dataset: {', '.join(missing_cols)}"}
        
        if target_feature not in current_data.columns:
            print(f"ERROR: Target column '{target_feature}' not in dataframe")
            eel.updateOptimizationProgress({
                "status": f"Error: Target column '{target_feature}' not found in dataset", 
                "progress": 0, 
                "error": True
            })
            return {"error": f"Target column '{target_feature}' not found in dataset"}
        
        print("Sending initial progress update")
        eel.updateOptimizationProgress({"status": "Analyzing ONNX model structure...", "progress": 5})
        
        
        model_info = analyze_onnx_model(current_model)
        print(f"Extracted model info: {model_info}")
        
        
        if not model_info or 'error' in model_info:
            eel.updateOptimizationProgress({
                "status": f"Could not analyze ONNX model structure: {model_info.get('error', 'Unknown error')}. Using default architectures.",
                "progress": 10
            })
            
            base_layers = 1
            base_neurons = 64
        else:
            
            base_layers = model_info.get('layers', 1)
            base_neurons = model_info.get('neurons', 64)
            
            eel.updateOptimizationProgress({
                "status": f"Found base architecture: {base_layers} layers with {base_neurons} neurons per layer",
                "progress": 15
            })
        
        
        print(f"Preparing data for features: {input_features}")
        X = current_data[input_features].values.astype(np.float32)
        print(f"X shape: {X.shape}")
        y = current_data[target_feature].values.astype(np.float32).reshape(-1, 1)
        print(f"y shape: {y.shape}")
        
        
        print("Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        
        print("Converting to PyTorch tensors")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        
        print("Creating dataset and dataloader")
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        print(f"DataLoader created with {len(train_loader)} batches")
        
        
        
        neurons_options = [
            max(16, base_neurons // 2),  
            base_neurons,                
            base_neurons * 2             
        ]
        
        layers_options = [
            max(1, base_layers - 1),  
            base_layers,              
            base_layers + 1           
        ]
        
        print(f"Testing configurations around base architecture: {len(neurons_options)*len(layers_options)} total combinations")
        eel.updateOptimizationProgress({
            "status": f"Testing {len(neurons_options)*len(layers_options)} variations of the model architecture",
            "progress": 20
        })
        
        
        results = []
        
        
        def create_model(input_size, hidden_size, num_layers):
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                
            layers.append(nn.Linear(hidden_size, 1))
            
            return nn.Sequential(*layers)
        
        
        total_iterations = len(neurons_options) * len(layers_options)
        
        
        best_loss = float('inf')
        current_iteration = 0
        
        for hidden_size in neurons_options:
            for num_layers in layers_options:
                
                current_iteration += 1
                progress_percent = 20 + int((current_iteration / total_iterations) * 60)  
                
                
                eel.updateOptimizationProgress({
                    "status": f"Testing {num_layers} layers with {hidden_size} neurons ({current_iteration}/{total_iterations})",
                    "progress": progress_percent
                })
                
                
                model = create_model(X_train.shape[1], hidden_size, num_layers)
                
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                
                train_losses = []
                
                for epoch in range(max_epochs):
                    model.train()
                    epoch_loss = 0
                    
                    for batch_X, batch_y in train_loader:
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    
                    avg_loss = epoch_loss / len(train_loader)
                    train_losses.append(avg_loss)
                    
                    
                    if epoch % 2 == 0:
                        eel.updateOptimizationProgress({
                            "status": f"Config: {num_layers} layers, {hidden_size} neurons - Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}",
                            "progress": progress_percent
                        })
                
                
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor).item()
                    
                    
                    test_predictions = test_outputs.numpy()
                    test_r2 = r2_score(y_test, test_predictions)
                
                
                result = {
                    "neurons": hidden_size,
                    "layers": num_layers,
                    "final_loss": train_losses[-1],
                    "test_loss": test_loss,
                    "r2_score": test_r2,
                    "training_curve": train_losses
                }
                
                results.append(result)
                
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_config = f"{num_layers} layers with {hidden_size} neurons"
                    
                    
                    eel.updateOptimizationProgress({
                        "status": f"New best: {best_config} (Loss: {best_loss:.4f})",
                        "progress": progress_percent
                    })
        
        
        best_result = min(results, key=lambda x: x["test_loss"])
        
        
        eel.updateOptimizationProgress({
        "status": f"Evaluation complete. Best architecture: {best_config} (Loss: {best_loss:.4f})",
        "progress": 80
        })
        
        
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
        
        torch.onnx.export(
            best_model, 
            dummy_input, 
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        
        
        eel.updateOptimizationProgress({
        "status": "Performing final training of optimal architecture...",
        "progress": 90
        })
        
        
        comparison = "optimized" if best_loss < base_neurons else "similar to original"
        
        print("Optimization complete!")
        print(f"Original model had approximately {base_layers} layers with {base_neurons} neurons")
        print(f"Best configuration: {best_result['layers']} layers with {best_result['neurons']} neurons ({comparison})")
        print(f"Test loss: {best_result['test_loss']}, R² score: {best_result['r2_score']}")
        
        return {
            "results": results,
            "best_config": {
                "neurons": best_result["neurons"],
                "layers": best_result["layers"],
                "test_loss": best_result["test_loss"],
                "r2_score": best_result["r2_score"],
                "original": {
                    "neurons": base_neurons,
                    "layers": base_layers
                }
            },
            "model_path": onnx_path
        }
        
    except Exception as e:
        print(f"ERROR in optimization: {str(e)}")
        traceback.print_exc()
        
        eel.updateOptimizationProgress({
            "status": f"Error: {str(e)}",
            "progress": 0,
            "error": True
        })
        return {"error": str(e)}


def analyze_onnx_model(model):
    
    try:
        if not model:
            return {"error": "No model provided"}
            
        graph = model.graph
        
        
        linear_ops = [node for node in graph.node if node.op_type in ['Gemm', 'MatMul']]
        num_layers = len(linear_ops) - 1  
        
        
        neurons = []
        for node in linear_ops:
            
            for input_name in node.input[1:]:  
                for init in graph.initializer:
                    if init.name == input_name:
                        
                        dims = [dim for dim in init.dims]
                        if len(dims) >= 2:
                            neurons.append(dims[0])  
        
        
        avg_neurons = int(np.median(neurons)) if neurons else 64
        
        return {
            "layers": max(1, num_layers),
            "neurons": avg_neurons,
            "ops": [node.op_type for node in graph.node]
        }
    except Exception as e:
        print(f"Error analyzing ONNX model: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

@eel.expose
def download_optimized_model(path):
    
    try:
        with open(path, "rb") as f:
            model_bytes = f.read()
        
        return {
            "base64": base64.b64encode(model_bytes).decode("utf-8"),
            "filename": "optimized_model.onnx"
        }
    except Exception as e:
        return {"error": str(e)}

screen_width, screen_height = pyautogui.size()


if __name__ == "__main__":
    try:
        
        eel_kwargs = {
            "size": (screen_width, screen_height),
            "port": 8000
        }
        
        
        if os.name == "nt":
            eel.start("index.html", mode="chrome", **eel_kwargs)
        
        
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()