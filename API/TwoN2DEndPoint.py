# TO DO -- Api end points:
# Add protection against spam
# Add api key based access

# run wtih uvicorn --app-dir . TwoN2DEndPoint:app --reload

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import onnx
import pandas as pd
import json
import base64
import os

from TwoN2D import (
    load_onnx_model,
    load_csv_data,
    find_optimal_architecture,
    download_optimized_model
)

app = FastAPI()

# Allow requests from your frontend (adjust if deploying)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

message_queues = {}
models = {}
data = {}
optimized = {}

@app.get("/")
def root():
    return {"message": "2N2D API is running"}

#Data handling

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...), session_id : str = Header(...)):

    if session_id in models and models[session_id] is not None:
        os.remove(models[session_id])

    contents = await file.read()
    base64_str = base64.b64encode(contents).decode('utf-8')
    result = load_onnx_model(base64_str)
    models[session_id] = result["path"]

    return JSONResponse(content=result)


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), session_id : str = Header(...)):

    contents = await file.read()
    base64_str = base64.b64encode(contents).decode('utf-8')
    result = load_csv_data(base64_str, file.filename)
    data[session_id] = result["path"]

    return JSONResponse(content=result)


#Optimization service

@app.post("/optimize")
async def optimize(request: dict, session_id : str = Header(...)):
    input_features = request.get("input_features")
    target_feature = request.get("target_feature")
    epochs = request.get("max_epochs")

    if session_id not in models:
        return "NO MODEL FOR USER"
    if session_id not in message_queues:
        message_queues[session_id] = []

    def status_callback(message):
        message_queues[session_id].append(message)


    result = find_optimal_architecture(
        current_model=onnx.load(models[session_id]),
        current_data=pd.read_csv(data[session_id]),
        input_features=input_features,
        target_feature=target_feature,
        status_callback=status_callback,
        max_epochs=epochs,
    )
    
    if "model_path" not in result:
        return JSONResponse(content=result, status_code=400)

    optimized[session_id] = result["model_path"]
    return JSONResponse(content=result)

@app.get("/optimization-status/{session_id}")
async def stream_status(session_id: str, request: Request):
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            if queue:
                message = queue.pop(0)
                yield f"data: {json.dumps(message)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download-optimized")
def download_optimized(session_id : str = Header(...)):
    result = download_optimized_model(optimized[session_id])
    return JSONResponse(content=result)


@app.post("/headerTest")
def headerTest(session_id : str = Header(...)):
    return {"session_id":session_id}