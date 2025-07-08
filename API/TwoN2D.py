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
import neat
import multiprocessing
import random
import traceback
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from Data import (analyze_csv_data)
from FileHandler import (createTempFile)

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    logging.warning("DEAP not available. Install with: pip install deap")
try:
    import onnxoptimizer
    HAS_ONNX_OPTIMIZER = True
except ImportError:
    HAS_ONNX_OPTIMIZER = False
    logging.warning("ONNX optimizer not available. Install with: pip install onnxoptimizer")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_debug.log"),
        logging.StreamHandler()
    ]
)

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

class UniversalArchitectureModel(nn.Module):
    """Universal PyTorch model that can replicate and enhance any ONNX architecture"""
    def __init__(self, architecture_info, enhancement_factor=1.0, actual_input_size=None, actual_output_size=None):
        super().__init__()
        self.architecture_info = architecture_info
        self.enhancement_factor = max(1.0, enhancement_factor)  
        self.actual_input_size = actual_input_size or architecture_info['input_size']
        self.actual_output_size = actual_output_size or architecture_info['output_size']
        self.needs_input_adapter = actual_input_size is not None and actual_input_size != architecture_info['input_size']
        self.needs_output_adapter = actual_output_size is not None and actual_output_size != architecture_info['output_size']
        if self.needs_input_adapter:
            self.input_flatten = nn.Flatten()
            self.input_adapter = nn.Linear(actual_input_size, architecture_info['input_size'])
            self.original_input_shape = None  
            
        if self.needs_output_adapter:
            self.output_adapter = nn.Linear(architecture_info['output_size'], actual_output_size)
        self._build_universal_architecture()
    
    def _build_universal_architecture(self):
        """Build PyTorch architecture from universal ONNX analysis"""
        arch_info = self.architecture_info
        layers = nn.ModuleList()
        if self.enhancement_factor > 1.0:
            enhanced_layers = max(1, int(arch_info.get('num_layers', 1) * self.enhancement_factor))
            enhanced_width = max(arch_info['input_size'], int(arch_info.get('max_width', 64) * self.enhancement_factor))
        else:
            enhanced_layers = arch_info.get('num_layers', 1)
            enhanced_width = arch_info.get('max_width', 64)
        current_size = arch_info['input_size']       
        if arch_info.get('has_lstm', False):
            if self.enhancement_factor > 1.0:
                lstm_hidden = max(arch_info.get('lstm_hidden_size', 64), int(arch_info.get('lstm_hidden_size', 64) * self.enhancement_factor))
                lstm_layers = max(arch_info.get('lstm_layers', 1), int(arch_info.get('lstm_layers', 1) * self.enhancement_factor))
            else:
                lstm_hidden = arch_info.get('lstm_hidden_size', 64)
                lstm_layers = arch_info.get('lstm_layers', 1)
            self.lstm = nn.LSTM(
                input_size=current_size,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.1 if lstm_layers > 1 and self.enhancement_factor > 1.0 else 0,
                bidirectional=arch_info.get('lstm_bidirectional', False)
            )
            lstm_output_size = lstm_hidden * (2 if arch_info.get('lstm_bidirectional', False) else 1)
            current_size = lstm_output_size
            self.lstm_return_sequences = arch_info.get('lstm_return_sequences', False)
            post_lstm_layers = arch_info.get('post_lstm_layers', 0)
            if post_lstm_layers > 0:
                if self.enhancement_factor > 1.0:
                    post_layers = max(1, int(post_lstm_layers * self.enhancement_factor))
                else:
                    post_layers = post_lstm_layers
                
                for i in range(post_layers):
                    if i == post_layers - 1:
                        layers.append(nn.Linear(current_size, arch_info['output_size']))
                    else:
                        next_size = max(enhanced_width // (i + 2), arch_info['output_size'])
                        layers.append(nn.Linear(current_size, next_size))
                        layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                        current_size = next_size
            else:
                layers.append(nn.Linear(current_size, arch_info['output_size']))
        elif arch_info.get('has_conv', False):
            conv_channels = arch_info.get('conv_channels', [32, 64])
            conv_type = '2d' if arch_info.get('has_conv2d', False) else '1d'  
            if self.enhancement_factor > 1.0:
                enhanced_channels = [max(c, int(c * self.enhancement_factor)) for c in conv_channels]
                if self.enhancement_factor > 1.2:
                    enhanced_channels.append(max(enhanced_channels[-1], int(enhanced_channels[-1] * 1.5)))
            else:
                enhanced_channels = conv_channels[:]
            
            
            input_channels = arch_info.get('input_channels', 1)
            if input_channels <= 0:
                input_channels = 1  
            
            prev_channels = input_channels
            conv_layers = []
            
            for i, channels in enumerate(enhanced_channels):
                if conv_type == '2d':
                    conv_layers.append(nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1))
                    
                    
                    if arch_info.get('has_normalization', False):
                        conv_layers.append(nn.BatchNorm2d(channels))
                    
                    conv_layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    
                    
                    if i % 2 == 1 or i == len(enhanced_channels) - 1:
                        
                        if i == len(enhanced_channels) - 1:
                            
                            conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                        else:
                            
                            conv_layers.append(nn.MaxPool2d(2, 2))
                else:  
                    conv_layers.append(nn.Conv1d(prev_channels, channels, kernel_size=3, padding=1))
                    
                    
                    if arch_info.get('has_normalization', False):
                        conv_layers.append(nn.BatchNorm1d(channels))
                    
                    conv_layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    
                    
                    if i % 2 == 1 or i == len(enhanced_channels) - 1:
                        if i == len(enhanced_channels) - 1:
                            
                            conv_layers.append(nn.AdaptiveAvgPool1d(1))
                        else:
                            
                            conv_layers.append(nn.MaxPool1d(2, 2))
                
                prev_channels = channels
            
            layers.extend(conv_layers)
            flattened_size = prev_channels  
            layers.append(nn.Flatten())
            if arch_info.get('post_conv_layers', 0) > 0:
                dense_sizes = arch_info.get('dense_layer_sizes', [enhanced_width])
                current_size = flattened_size
                for dense_size in dense_sizes:
                    if self.enhancement_factor > 1.0:
                        enhanced_dense_size = max(dense_size, int(dense_size * self.enhancement_factor))
                    else:
                        enhanced_dense_size = dense_size
                    layers.append(nn.Linear(current_size, enhanced_dense_size))
                    layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    current_size = enhanced_dense_size
                layers.append(nn.Linear(current_size, arch_info['output_size']))
            else:
                layers.append(nn.Linear(flattened_size, arch_info['output_size']))
            self.conv_type = conv_type
        
        else:
            layer_sizes = arch_info.get('layer_sizes', [enhanced_width] * enhanced_layers)
            if self.enhancement_factor > 1.0:
                enhanced_sizes = []
                for size in layer_sizes:
                    enhanced_size = max(size, int(size * self.enhancement_factor))
                    enhanced_sizes.append(enhanced_size)
                if self.enhancement_factor > 1.5:
                    extra_layers = int((self.enhancement_factor - 1.0) * 2)
                    for _ in range(extra_layers):                        enhanced_sizes.insert(-1, max(enhanced_sizes[-2] // 2, arch_info['output_size']))
            else:
                enhanced_sizes = layer_sizes[:]
                
            
            for i, size in enumerate(enhanced_sizes):
                if i == len(enhanced_sizes) - 1:
                    layers.append(nn.Linear(current_size, arch_info['output_size']))
                else:
                    layers.append(nn.Linear(current_size, size))
                    
                    
                    if arch_info.get('has_normalization', False) and self.enhancement_factor > 1.0:
                        layers.append(nn.BatchNorm1d(size))
                    
                    layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    if self.enhancement_factor > 1.3:
                        layers.append(nn.Dropout(0.1))
                    current_size = size
        
        self.layers = layers
        
        
        self.arch_type = 'lstm' if arch_info.get('has_lstm', False) else 'conv' if arch_info.get('has_conv', False) else 'feedforward'
        
        logging.info(f"Universal architecture built: {self.arch_type} | Enhancement: {self.enhancement_factor:.2f} | Layers: {len([l for l in layers if isinstance(l, nn.Linear)])}")
    
    def _get_activation(self, activation_type):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),  
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
            'relu6': nn.ReLU6(),
            'hardtanh': nn.Hardtanh(),
            'hardsigmoid': nn.Hardsigmoid(),
            'hardswish': nn.Hardswish(),
            'mish': nn.Mish(),  
        }
        return activations.get(activation_type.lower(), nn.ReLU())
    def forward(self, x):
        original_shape = x.shape
        if self.needs_input_adapter:
            if len(x.shape) > 2:
                self.original_input_shape = x.shape
                x = self.input_flatten(x)
            x = self.input_adapter(x)
        if self.arch_type == 'lstm':
            if hasattr(self, 'lstm'):
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)
                elif len(x.shape) > 3:
                    batch_size = x.size(0)
                    x = x.view(batch_size, -1, x.size(-1))
                
                lstm_out, (h_n, c_n) = self.lstm(x)
                if hasattr(self, 'lstm_return_sequences') and self.lstm_return_sequences:
                    x = lstm_out.contiguous().view(lstm_out.size(0), -1)
                else:
                    if len(lstm_out.shape) == 3:
                        x = lstm_out[:, -1, :]  
                    else:
                        x = lstm_out
            for layer in self.layers:
                x = layer(x)
        
        elif self.arch_type == 'conv':
            conv_type = getattr(self, 'conv_type', '1d')
            
            if conv_type == '2d':
                if len(x.shape) == 2:
                    batch_size, features = x.shape
                    img_size = int(np.sqrt(features))
                    if img_size * img_size == features:
                        x = x.view(batch_size, 1, img_size, img_size)
                    else:
                        x = x.view(batch_size, 1, 1, features)
                elif len(x.shape) == 3:
                    x = x.unsqueeze(-1)  
            else:  
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  
                elif len(x.shape) > 3:
                    batch_size = x.size(0)
                    x = x.view(batch_size, 1, -1)
            for layer in self.layers:
                x = layer(x)
        
        else:  
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            
            for layer in self.layers:
                x = layer(x)
        if self.needs_output_adapter:
            x = self.output_adapter(x)
            if hasattr(self, 'output_reshape_needed') and self.output_reshape_needed:
                pass
        
        return x

class NEATPytorchModel(nn.Module):
    """Universal NEAT PyTorch model that enhances any ONNX architecture"""
    def __init__(self, genome, config, input_size, output_size, base_model_info):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.base_model_info = base_model_info
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        self._build_enhanced_universal_architecture(genome)
        
    def _build_enhanced_universal_architecture(self, genome):
        """Build enhanced PyTorch architecture using universal approach"""
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        total_nodes = len(genome.nodes)
        baseline_connections = self.input_size + self.output_size  
        actual_connections = len(enabled_connections)
        connection_ratio = actual_connections / max(baseline_connections, 1)
        node_ratio = total_nodes / max(self.input_size + self.output_size + 2, 1)
        complexity_score = (connection_ratio * 2.0) + (node_ratio * 1.5)
        min_enhancement = 1.25
        max_enhancement = min(3.0, 1.0 + complexity_score)
        enhancement_factor = max(min_enhancement, max_enhancement)
        
        logging.info(f"NEAT Enhancement Factor: {enhancement_factor:.2f} | "
                    f"Connections: {actual_connections} | Nodes: {total_nodes}")
        self.enhanced_model = UniversalArchitectureModel(
            architecture_info=self.base_model_info,
            enhancement_factor=enhancement_factor,
            actual_input_size=self.input_size,
            actual_output_size=self.output_size
        )
        original_params = self._count_parameters_from_info(self.base_model_info, 1.0)
        enhanced_params = self._count_parameters_from_info(self.base_model_info, enhancement_factor)
        
        if enhanced_params <= original_params:
            logging.warning(f"NEAT: Parameter count not enhanced! {enhanced_params} <= {original_params}")
        else:
            logging.info(f"NEAT: Parameters enhanced from {original_params} to {enhanced_params}")
    def _count_parameters_from_info(self, arch_info, factor):
        """Accurate parameter count estimation for universal architectures"""
        total_params = 0
        
        input_size = arch_info.get('input_size', 64)
        output_size = arch_info.get('output_size', 1)
        max_width = arch_info.get('max_width', 64)
        num_layers = arch_info.get('num_layers', 1)
        if factor > 1.0:
            max_width = max(max_width, int(max_width * factor))
            num_layers = max(num_layers, int(num_layers * factor))
        
        if arch_info.get('has_lstm', False):
            lstm_hidden = arch_info.get('lstm_hidden_size', 64)
            lstm_layers = arch_info.get('lstm_layers', 1)
            
            if factor > 1.0:
                lstm_hidden = max(lstm_hidden, int(lstm_hidden * factor))
                lstm_layers = max(lstm_layers, int(lstm_layers * factor))
            for layer in range(lstm_layers):
                layer_input_size = input_size if layer == 0 else lstm_hidden
                total_params += 4 * layer_input_size * lstm_hidden
                total_params += 4 * lstm_hidden * lstm_hidden
                total_params += 4 * lstm_hidden
            if arch_info.get('lstm_bidirectional', False):
                total_params *= 2
            post_layers = arch_info.get('post_lstm_layers', 0)
            if post_layers > 0:
                current_size = lstm_hidden * (2 if arch_info.get('lstm_bidirectional', False) else 1)
                layer_sizes = arch_info.get('layer_sizes', [max_width] * (post_layers - 1))
                
                for size in layer_sizes:
                    if factor > 1.0:
                        size = max(size, int(size * factor))
                    total_params += current_size * size + size  
                    current_size = size
                total_params += current_size * output_size + output_size
            else:
                lstm_output_size = lstm_hidden * (2 if arch_info.get('lstm_bidirectional', False) else 1)
                total_params += lstm_output_size * output_size + output_size
                
        elif arch_info.get('has_conv', False):
            conv_channels = arch_info.get('conv_channels', [32, 64])
            input_channels = arch_info.get('input_channels', 1)
            
            if factor > 1.0:
                conv_channels = [max(c, int(c * factor)) for c in conv_channels]
            
            prev_channels = input_channels
            conv_type = '2d' if arch_info.get('has_conv2d', False) else '1d'
            for channels in conv_channels:
                if conv_type == '2d':
                    total_params += 3 * 3 * prev_channels * channels + channels
                else:
                    total_params += 3 * prev_channels * channels + channels
                prev_channels = channels
            post_conv_layers = arch_info.get('post_conv_layers', 0)
            if post_conv_layers > 0:
                current_size = prev_channels
                dense_sizes = arch_info.get('dense_layer_sizes', [max_width])
                
                for size in dense_sizes:
                    if factor > 1.0:
                        size = max(size, int(size * factor))
                    total_params += current_size * size + size
                    current_size = size
                
                total_params += current_size * output_size + output_size
            else:
                total_params += prev_channels * output_size + output_size
                
        else:
            layer_sizes = arch_info.get('layer_sizes', [max_width] * (num_layers - 1))
            
            if factor > 1.0:
                layer_sizes = [max(size, int(size * factor)) for size in layer_sizes]
                if factor > 1.5:
                    extra_layers = int((factor - 1.0) * 2)
                    for _ in range(extra_layers):
                        layer_sizes.insert(-1, max(layer_sizes[-2] // 2, output_size))
            current_size = input_size
            for size in layer_sizes:
                total_params += current_size * size + size 
                current_size = size
            total_params += current_size * output_size + output_size
        
        return int(total_params)
    
    def forward(self, x):
        return self.enhanced_model(x)

class StructurePreservingModel(nn.Module):
    """
    Enhanced PyTorch model that preserves the structure of the original ONNX model.
    Now uses UniversalArchitectureModel as backend for better compatibility.
    Supports both feedforward and LSTM architectures with input/output adapters.
    """
    def __init__(self, model_type, input_size, hidden_size, num_layers, output_size, 
                 actual_input_size=None, actual_output_size=None):
        super(StructurePreservingModel, self).__init__()
        architecture_info = {
            'input_size': input_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'max_width': hidden_size,
            'layer_sizes': [hidden_size] * num_layers,
            'has_lstm': model_type == 'lstm',
            'has_conv': False,
            'activation_type': 'relu',
            'lstm_hidden_size': hidden_size if model_type == 'lstm' else None,
            'lstm_layers': num_layers if model_type == 'lstm' else None,
            'post_lstm_layers': 1 if model_type == 'lstm' else 0,
        }
        self.universal_model = UniversalArchitectureModel(
            architecture_info=architecture_info,
            enhancement_factor=1.0,  
            actual_input_size=actual_input_size,
            actual_output_size=actual_output_size
        )
        self.needs_input_adapter = actual_input_size is not None and actual_input_size != input_size
        self.input_size = actual_input_size if self.needs_input_adapter else input_size
        self.model_type = model_type
        self.needs_output_adapter = actual_output_size is not None and actual_output_size != output_size
    
    def forward(self, x):
        return self.universal_model(x)

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
        info = analyze_csv_data(df)


        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "filename": filename,
            "missing_values": df.isna().sum().to_dict(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        result = {"data": df.head(50).to_dict('records'), "summary": summary, "results": info}
        result = make_json_serializable(result)
        return result
    except Exception as e:
        logging.exception("Failed to load CSV data")
        return {"error": str(e)}

def analyze_onnx_model_universal(model_bytes):
    """Universal ONNX model analyzer that handles any architecture"""
    try:
        tempFilePath = createTempFile(model_bytes, ".onnx")
        model = onnx.load(tempFilePath)
        graph = model.graph
        input_shapes = {}
        for inp in graph.input:
            shape = []
            if hasattr(inp.type.tensor_type, 'shape'):
                for dim in inp.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else None)
            input_shapes[inp.name] = shape
        
        output_shapes = {}
        for out in graph.output:
            shape = []
            if hasattr(out.type.tensor_type, 'shape'):
                for dim in out.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else None)
            output_shapes[out.name] = shape
        op_counts = {}
        op_sequence = []
        layer_dimensions = []
        activation_types = set()
        
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
            op_sequence.append(node.op_type)
            if node.op_type in ['Relu', 'Sigmoid', 'Tanh', 'LeakyRelu', 'Gelu', 'Swish']:
                activation_types.add(node.op_type.lower())
        weight_dimensions = {}
        for init in graph.initializer:
            weight_dimensions[init.name] = list(init.dims)
        has_lstm = op_counts.get('LSTM', 0) > 0
        has_gru = op_counts.get('GRU', 0) > 0
        has_rnn = op_counts.get('RNN', 0) > 0
        has_conv1d = op_counts.get('Conv', 0) > 0 and any(len(weight_dimensions.get(init.name, [])) == 3 for init in graph.initializer if 'conv' in init.name.lower())
        has_conv2d = op_counts.get('Conv', 0) > 0 and any(len(weight_dimensions.get(init.name, [])) == 4 for init in graph.initializer if 'conv' in init.name.lower())
        has_conv = op_counts.get('Conv', 0) > 0
        has_attention = any(op in op_counts for op in ['Attention', 'MultiHeadAttention'])
        has_normalization = any(op in op_counts for op in ['BatchNormalization', 'LayerNormalization', 'InstanceNormalization'])
        input_channels = 1  
        if has_conv:
            for node in graph.node:
                if node.op_type == 'Conv':
                    for inp in node.input:
                        if inp in weight_dimensions:
                            dims = weight_dimensions[inp]
                            if len(dims) >= 3:  
                                input_channels = dims[1] 
                                break
                    break
        linear_ops = ['Gemm', 'MatMul', 'Linear']
        linear_layers = sum(op_counts.get(op, 0) for op in linear_ops)
        input_size = 0
        output_size = 1
        max_width = 64
        layer_sizes = []
        if input_shapes:
            first_input = list(input_shapes.values())[0]
            if first_input and len(first_input) > 1:
                if first_input[-1] is not None and first_input[-1] > 0:
                    input_size = first_input[-1]
                elif len(first_input) > 1 and first_input[1] is not None and first_input[1] > 0:
                    input_size = first_input[1]
        if output_shapes:
            first_output = list(output_shapes.values())[0]
            if first_output and len(first_output) > 0:
                if first_output[-1] is not None and first_output[-1] > 0:
                    output_size = first_output[-1]
                elif len(first_output) > 1 and first_output[1] is not None and first_output[1] > 0:
                    output_size = first_output[1]
        linear_weights = []
        extracted_layer_sizes = []
        for node in graph.node:
            if node.op_type in linear_ops:
                for inp in node.input:
                    if inp in weight_dimensions:
                        dims = weight_dimensions[inp]
                        if len(dims) >= 2:
                            linear_weights.append(dims)
                            output_dim = dims[0]
                            input_dim = dims[1]
                            extracted_layer_sizes.append(output_dim)
                            max_width = max(max_width, output_dim, input_dim)
                            if input_size <= 0 and len(extracted_layer_sizes) == 1:
                                input_size = input_dim
        if extracted_layer_sizes:
            layer_sizes = extracted_layer_sizes[:-1] if len(extracted_layer_sizes) > 1 else [max_width]
            if extracted_layer_sizes and output_size == 1:
                output_size = extracted_layer_sizes[-1]
        lstm_hidden_size = 64
        lstm_layers = 1
        if has_lstm or has_gru or has_rnn:
            for node in graph.node:
                if node.op_type in ['LSTM', 'GRU', 'RNN']:
                    for inp in node.input:
                        if inp in weight_dimensions:
                            dims = weight_dimensions[inp]
                            if len(dims) >= 2:
                                if node.op_type == 'LSTM':
                                    potential_hidden = dims[0] // 4
                                    if potential_hidden > 0:
                                        lstm_hidden_size = max(lstm_hidden_size, potential_hidden)
                                else:
                                    lstm_hidden_size = max(lstm_hidden_size, dims[0])
                                break
                    for attr in node.attribute:
                        if attr.name == 'hidden_size':
                            lstm_hidden_size = max(lstm_hidden_size, attr.i)
                        elif attr.name == 'num_layers' or attr.name == 'direction':
                            if attr.name == 'num_layers':
                                lstm_layers = max(lstm_layers, attr.i)
        conv_channels = []
        if has_conv:
            for node in graph.node:
                if node.op_type == 'Conv':
                    for inp in node.input:
                        if inp in weight_dimensions:
                            dims = weight_dimensions[inp]
                            if len(dims) >= 3:
                                conv_channels.append(dims[0])
        num_layers = max(1, len([op for op in op_sequence if op in linear_ops + ['LSTM', 'GRU', 'RNN', 'Conv']]))
        primary_activation = 'relu'
        if activation_types:
            activation_priority = ['gelu', 'swish', 'leakyrelu', 'relu', 'tanh', 'sigmoid']
            for act in activation_priority:
                if act in activation_types:
                    primary_activation = act
                    break
        post_recurrent_layers = 0
        if has_lstm or has_gru or has_rnn:
            recurrent_found = False
            for op in op_sequence:
                if op in ['LSTM', 'GRU', 'RNN']:
                    recurrent_found = True
                elif recurrent_found and op in linear_ops:
                    post_recurrent_layers += 1
        if input_size <= 0:
            input_size = max(layer_sizes[0] if layer_sizes else 64, 1)
        if not layer_sizes:
            layer_sizes = [max_width] * max(1, num_layers - 1)
        architecture_info = {
            'input_size': input_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'max_width': max_width,
            'layer_sizes': layer_sizes,
            'has_lstm': has_lstm,
            'has_gru': has_gru,
            'has_rnn': has_rnn,
            'has_conv': has_conv,
            'has_conv1d': has_conv1d,
            'has_conv2d': has_conv2d,
            'has_attention': has_attention,
            'has_normalization': has_normalization,
            'activation_type': primary_activation,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_layers': lstm_layers,
            'lstm_bidirectional': False,  
            'lstm_return_sequences': False,
            'post_lstm_layers': post_recurrent_layers,
            'conv_channels': conv_channels,
            'input_channels': input_channels,
            'post_conv_layers': 1 if has_conv and linear_layers > 0 else 0,
            'dense_layer_sizes': [max_width] if linear_layers > 0 else [],  
            'op_counts': op_counts,
            'op_sequence': op_sequence,
            'model_type': 'lstm' if (has_lstm or has_gru or has_rnn) else 'conv' if has_conv else 'feedforward',
            'hidden_size': max_width,
            'is_recurrent': has_lstm or has_gru or has_rnn,
            'layers': num_layers,
            'neurons': max_width
        }
        
        logging.info(f"Universal ONNX Analysis: {architecture_info['model_type']} | "
                    f"Layers: {num_layers} | Max Width: {max_width} | "
                    f"Input: {input_size} | Output: {output_size}")
        logging.info(f"Operators: {dict(list(op_counts.items())[:5])}...")
        
        os.remove(tempFilePath)
        return architecture_info
        
    except Exception as e:
        logging.exception("Error analyzing ONNX model universally")
        if 'tempFilePath' in locals():
            try:
                os.remove(tempFilePath)
            except:
                pass
        return {
            'input_size': 64,
            'output_size': 1,
            'num_layers': 2,
            'max_width': 128,
            'layer_sizes': [128, 64],
            'has_lstm': False,
            'has_gru': False,
            'has_rnn': False,
            'has_conv': False,
            'has_conv1d': False,
            'has_conv2d': False,
            'has_attention': False,
            'has_normalization': False,
            'activation_type': 'relu',
            'lstm_hidden_size': 64,
            'lstm_layers': 1,
            'post_lstm_layers': 0,
            'conv_channels': [],
            'op_counts': {},
            'op_sequence': [],
            'model_type': 'feedforward',
            'hidden_size': 128,
            'is_recurrent': False,
            'layers': 1,
            'neurons': 64
        }

def analyze_onnx_model(model_bytes):
    """Original function name maintained for backward compatibility"""
    return analyze_onnx_model_universal(model_bytes)

def optimize_with_neat(X_train, y_train, input_size, output_size, config_path, base_model_info, generations=20):
    """Optimize model architecture using NEAT while preserving base structure"""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            try:
                net = NEATPytorchModel(genome, config, input_size, output_size, base_model_info)
                optimizer = optim.Adam(net.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                net.train()
                for epoch in range(5):  
                    epoch_loss = 0
                    batch_count = 0
                    for i in range(0, len(X_train), 32):
                        end_idx = min(i + 32, len(X_train))
                        X_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32)
                        y_batch = torch.tensor(y_train[i:end_idx], dtype=torch.float32)
                        
                        optimizer.zero_grad()
                        outputs = net(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                net.eval()
                total_loss = 0
                eval_batches = 0
                with torch.no_grad():
                    for i in range(0, len(X_train), 32):
                        end_idx = min(i + 32, len(X_train))
                        X_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32)
                        y_batch = torch.tensor(y_train[i:end_idx], dtype=torch.float32)
                        
                        outputs = net(X_batch)
                        loss = criterion(outputs, y_batch)
                        total_loss += loss.item()
                        eval_batches += 1
                avg_loss = total_loss / max(eval_batches, 1)
                genome.fitness = 1.0 / (avg_loss + 1e-5)
                if genome_id <= 5:
                    logging.info(f"NEAT Genome {genome_id}: Loss={avg_loss:.4f}, Fitness={genome.fitness:.4f}")
                
            except Exception as e:
                genome.fitness = 1e-6
                logging.warning(f"Error evaluating genome {genome_id}: {e}")
    winner = p.run(eval_genomes, generations)
    winner_model = NEATPytorchModel(winner, config, input_size, output_size, base_model_info)
    logging.info("Optimization complete. Visualization disabled to reduce dependencies.")
    
    return winner_model

def optimize_with_genetic_deap(X_train, y_train, input_size, output_size, base_model_info, population_size=20, generations=10):
    """DEAP-based genetic algorithm that enhances original model architecture"""
    if not HAS_DEAP:
        raise ImportError("DEAP library is required for genetic algorithm optimization. Please install it with: pip install deap")
    base_layers = base_model_info.get('num_layers', 1)
    base_neurons = base_model_info.get('hidden_size', 64)
    model_type = base_model_info.get('model_type', 'feedforward')
    
    logging.info(f"DEAP GA starting with base: {base_layers} layers, {base_neurons} neurons")
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    def create_individual():
        enhancement_factor = random.uniform(1.2, 3.0)
        pattern = random.randint(0, 3)
        return creator.Individual([enhancement_factor, pattern])
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_individual(individual):
        """Evaluate an individual using universal architecture"""
        try:
            enhancement_factor, architecture_pattern = individual
            enhancement_factor = max(1.2, enhancement_factor)
            enhanced_arch_info = base_model_info.copy()
            enhanced_arch_info['enhancement_pattern'] = architecture_pattern
            model = UniversalArchitectureModel(
                architecture_info=enhanced_arch_info,
                enhancement_factor=enhancement_factor,
                actual_input_size=input_size,
                actual_output_size=output_size
            )
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            for epoch in range(3):  
                total_loss = 0
                batch_count = 0
                for i in range(0, len(X_train), 32):
                    end_idx = min(i + 32, len(X_train))
                    X_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32)
                    y_batch = torch.tensor(y_train[i:end_idx], dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = nn.MSELoss()(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
            model.eval()
            with torch.no_grad():
                total_eval_loss = 0
                eval_batches = 0
                for i in range(0, len(X_train), 32):
                    end_idx = min(i + 32, len(X_train))
                    X_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32)
                    y_batch = torch.tensor(y_train[i:end_idx], dtype=torch.float32)
                    
                    outputs = model(X_batch)
                    loss = nn.MSELoss()(outputs, y_batch)
                    total_eval_loss += loss.item()
                    eval_batches += 1
                
                avg_loss = total_eval_loss / max(eval_batches, 1)
                fitness = 1.0 / (avg_loss + 1e-6)  
                
            return (fitness,)
        except Exception as e:
            logging.warning(f"Error evaluating DEAP individual: {e}")
            import traceback
            logging.warning(f"Traceback: {traceback.format_exc()}")
            return (1e-6,)  
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    def custom_mutate(individual):
        """Custom mutation that ensures enhancement constraints"""
        if random.random() < 0.3: 
            individual[0] = max(1.2, min(3.0, individual[0] + random.uniform(-0.3, 0.3)))
        if random.random() < 0.2:  
            individual[1] = random.randint(0, 3)
        return (individual,)
    
    toolbox.register("mutate", custom_mutate)
    population = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    best_fitness_history = []
    best_individual = None
    best_fitness = 0
    
    for gen in range(generations):
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        current_best = max(population, key=lambda x: x.fitness.values[0])
        if current_best.fitness.values[0] > best_fitness:
            best_fitness = current_best.fitness.values[0]
            best_individual = current_best[:]
        
        best_fitness_history.append(best_fitness)
        logging.info(f"DEAP Gen {gen+1}: Best Fitness = {best_fitness:.4f} "
                    f"(Enhancement={best_individual[0]:.2f}, Pattern={best_individual[1]})")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.3:  
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        population[:] = offspring
    enhancement_factor, pattern = best_individual
    enhanced_arch_info = base_model_info.copy()
    enhanced_arch_info['enhancement_pattern'] = pattern
    best_model = UniversalArchitectureModel(
        architecture_info=enhanced_arch_info,
        enhancement_factor=enhancement_factor,
        actual_input_size=input_size,
        actual_output_size=output_size
    )
    
    logging.info(f"DEAP final best: enhancement factor {enhancement_factor:.2f}, pattern {pattern}")
    logging.info(f"Universal architecture created for any ONNX model type")
    
    return best_model, best_fitness_history

def create_neat_config(input_size, output_size):
    """Create a NEAT configuration file for the given problem"""
    config_path = "neat_config.ini"
    
    config_content = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000
pop_size              = 20
reset_on_extinction   = False

[DefaultGenome]
num_inputs            = {input_size}
num_outputs           = {output_size}
num_hidden            = 0
feed_forward          = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
delay_init_mean         = 1.0
delay_init_stdev        = 0.0
delay_max_value         = 1.0
delay_min_value         = 0.0
delay_mutate_power      = 0.0
delay_mutate_rate       = 0.0
delay_replace_rate      = 0.0
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
node_add_prob           = 0.2
node_delete_prob        = 0.2

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path

def find_optimal_architecture(onnx_bytes, csv_bytes, input_features, target_feature, max_epochs=5, 
                            strategy='brute-force', generations=10, status_callback=None):
    """
    Enhanced version of the original function with new optimization strategies.
    Maintains backward compatibility while adding NEAT and DEAP genetic algorithms.
    """
    
    def send_status(message):
        if status_callback:
            status_callback(message)
        else:
            logging.info(message.get("status", str(message)))

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(4)  
        
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
        logging.info(f"Using optimization strategy: {strategy}")

        if strategy == 'brute-force':
            neurons_options = [max(16, base_neurons // 2), base_neurons]  
            layers_options = [max(1, base_layers - 1), base_layers]       

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
                                if outputs.shape[0] == batch_y.shape[0]:  
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
                                if outputs.shape[0] == batch_y.shape[0]:
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
                        "neurons": int(hidden_size), 
                        "layers": int(num_layers), 
                        "test_loss": float(test_loss), 
                        "r2_score": float(r2)
                    })
                    
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_result = {
                            "neurons": int(hidden_size), 
                            "layers": int(num_layers), 
                            "test_loss": float(test_loss), 
                            "r2_score": float(r2)
                        }
                        best_model = model
                        send_status({
                            "status": f"New best config: {num_layers} layers, {hidden_size} neurons",
                            "progress": progress_percent
                        })

            send_status({"status": "Final training of best model...", "progress": 90})
            
        elif strategy == 'neat':
            send_status({"status": "Starting NEAT optimization...", "progress": 20})
            config_path = create_neat_config(actual_input_size, actual_output_size)
            best_model = optimize_with_neat(
                X_train, y_train, 
                actual_input_size, actual_output_size,
                config_path, model_info, generations
            )
            send_status({"status": "Training best NEAT model with backpropagation...", "progress": 80})
            train_loader_neat = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32)
                ), 
                batch_size=32, 
                shuffle=True
            )
            
            optimizer = optim.Adam(best_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(max_epochs*2):
                best_model.train()
                for batch_X, batch_y in train_loader_neat:
                    optimizer.zero_grad()
                    outputs = best_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            best_result = {"strategy": "NEAT", "base_model": model_info}
            results = [{"strategy": "NEAT", "generations": generations}]
            
        elif strategy == 'genetic':
            send_status({"status": "Starting DEAP genetic optimization...", "progress": 20})
            
            best_model, history = optimize_with_genetic_deap(
                X_train, y_train,
                actual_input_size, actual_output_size,
                model_info, population_size=20, generations=generations
            )
            send_status({"status": "Final training of best genetic model...", "progress": 80})
            train_loader_genetic = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32)
                ), 
                batch_size=32, 
                shuffle=True
            )
            
            optimizer = optim.Adam(best_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(max_epochs*2):
                best_model.train()
                for batch_X, batch_y in train_loader_genetic:
                    optimizer.zero_grad()
                    outputs = best_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            best_result = {"strategy": "Genetic", "history": history}
            results = [{"strategy": "Genetic", "generations": generations, "fitness_history": history}]
            
        else:
            send_status({"status": f"Unknown strategy: {strategy}", "error": True})
            return {"error": f"Unknown strategy: {strategy}"}
        best_model = best_model.to("cpu")
        dummy_input = torch.randn(1, actual_input_size, device="cpu")
        onnx_path = os.path.join(tempfile.gettempdir(), "optimized_model.onnx")
        best_model.eval()
        torch.onnx.export(
            best_model, 
            dummy_input, 
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12,
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )
        onnx_model = onnx.load(onnx_path)
        ops_before = len(onnx_model.graph.node)
        node_types_before = {}
        for node in onnx_model.graph.node:
            node_types_before[node.op_type] = node_types_before.get(node.op_type, 0) + 1
        
        logging.info(f"ONNX model before optimization: {ops_before} operations")
        logging.info(f"Node types before: {dict(sorted(node_types_before.items(), key=lambda x: x[1], reverse=True)[:5])}")
        if HAS_ONNX_OPTIMIZER:
            try:
                optimization_passes = [
                    'eliminate_identity',
                    'eliminate_nop_transpose', 
                    'eliminate_nop_pad',
                    'eliminate_duplicate_initializer',
                    'eliminate_deadend',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_unsqueezes',
                    'eliminate_nop_reshape',
                    'eliminate_nop_dropout',
                    'fuse_add_bias_into_conv',
                    'eliminate_unused_initializer',
                    'fuse_matmul_add_bias_into_gemm',
                ]
                onnx_model = onnxoptimizer.optimize(onnx_model, optimization_passes)
                ops_after = len(onnx_model.graph.node)
                node_types_after = {}
                for node in onnx_model.graph.node:
                    node_types_after[node.op_type] = node_types_after.get(node.op_type, 0) + 1
                
                logging.info(f"✓ ONNX Optimizer applied successfully:")
                logging.info(f"  Operations: {ops_before} → {ops_after} (reduced by {ops_before - ops_after})")
                logging.info(f"  Node types after: {dict(sorted(node_types_after.items(), key=lambda x: x[1], reverse=True)[:5])}")
                logging.info(f"  Applied {len(optimization_passes)} optimization passes")
                excessive_ops = ['Slice', 'Unsqueeze', 'Concat', 'Squeeze', 'Reshape', 'Transpose']
                excessive_count_before = sum(node_types_before.get(op, 0) for op in excessive_ops)
                excessive_count_after = sum(node_types_after.get(op, 0) for op in excessive_ops)
                
                if excessive_count_after < excessive_count_before:
                    logging.info(f"  Excessive operations reduced: {excessive_count_before} → {excessive_count_after}")
                elif excessive_count_after > 10:
                    logging.warning(f"  Still {excessive_count_after} potentially excessive operations remaining")
                    
            except Exception as e:
                logging.warning(f"ONNX optimization failed: {e}")
                logging.info("Continuing with unoptimized model")
        else:
            logging.info("ONNX optimizer not available, skipping post-processing optimization")
        try:
            nodes_renamed = 0
            for i, node in enumerate(onnx_model.graph.node):
                if any(prefix in node.name for prefix in ['Universal', 'universal', '_layers_', '_adapter', 'enhanced_']):
                    base_name = node.op_type
                    existing_names = [n.name for n in onnx_model.graph.node if n != node]
                    if base_name in existing_names:
                        counter = 1
                        while f"{base_name}_{counter}" in existing_names:
                            counter += 1
                        node.name = f"{base_name}_{counter}"
                    else:
                        node.name = base_name
                    nodes_renamed += 1
            
            if nodes_renamed > 0:
                logging.info(f"✓ Post-export cleanup: Simplified {nodes_renamed} node names to operation types for cleaner visualization")
        except Exception as e:
            logging.warning(f"Node name cleanup failed: {e}")
        try:
            onnx.checker.check_model(onnx_model)
            onnx.save(onnx_model, onnx_path)
            logging.info("✓ Final ONNX model validation passed")
        except Exception as e:
            logging.error(f"ONNX model validation failed: {e}")
        final_ops = len(onnx_model.graph.node)
        final_node_types = {}
        for node in onnx_model.graph.node:
            final_node_types[node.op_type] = final_node_types.get(node.op_type, 0) + 1
        
        logging.info("=== Final ONNX Model Summary ===")
        logging.info(f"Total operations: {final_ops}")
        logging.info(f"Main operation types: {dict(sorted(final_node_types.items(), key=lambda x: x[1], reverse=True)[:10])}")
        logging.info("ONNX export and optimization complete")
        
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
