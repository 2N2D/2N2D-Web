# TO DO -- Api end points:
# Add protection against spam
# Add api key based access

# run wtih uvicorn --app-dir . TwoN2DEndPoint:app --reload --log-level debug

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import onnx
import pandas as pd
import json
from pydantic import BaseModel
import os
import concurrent.futures

from TwoN2D import (
    load_onnx_model,
    load_csv_data,
    find_optimal_architecture,
    download_optimized_model
)

from FileHandler import (
    getFileBinaryData,
    uploadFile
)


class FilePathRequest(BaseModel):
    filepath: str


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


@app.get("/")
def root():
    return {"message": "2N2D API is running"}


# data handling

@app.post("/upload-model")
async def upload_model(request: FilePathRequest, session_id: str = Header(...)):
    binary_data = await getFileBinaryData(request.filepath, "onnx")
    result = load_onnx_model(binary_data)

    return JSONResponse(content=result)


@app.post("/upload-csv")
async def upload_csv(request: FilePathRequest, session_id: str = Header(...)):
    binary_data = await getFileBinaryData(request.filepath, "csv")
    result = load_csv_data(binary_data, os.path.basename(request.filepath))
    return JSONResponse(content=result)


# Optimization service

@app.post("/optimize")
async def optimize(request: dict, session_id: str = Header(...)):
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    message_queues[session_id].append("Processing request...")
    input_features = request.get("input_features")
    target_feature = request.get("target_feature")
    epochs = request.get("max_epochs")
    sessionId = request.get("session_id")
    csv_path = request.get("csv_path")
    onnx_path = request.get("onnx_path")

    print(csv_path)

    message_queues[session_id].append({
        "status": "Downloading csv data from database...",
        "progress": 5
    })
    csv_binary = await getFileBinaryData(csv_path, "csv")

    message_queues[session_id].append({
        "status": "Downloading onnx data from database...",
        "progress": 10
    })
    onnx_binary = await getFileBinaryData(onnx_path, "onnx")

    def status_callback(message):
        message_queues[session_id].append(message)

    import concurrent.futures

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: find_optimal_architecture(
                onnx_bytes=onnx_binary,
                csv_bytes=csv_binary,
                input_features=input_features,
                target_feature=target_feature,
                status_callback=status_callback,
                max_epochs=epochs,
            )
        )

    if "model_path" not in result:
        return JSONResponse(content=result, status_code=400)

    filename = os.path.basename(result["model_path"])
    await uploadFile(result["model_path"], f"{session_id}/{sessionId}/{filename}")
    result["url"] = f"{session_id}/{sessionId}/{filename}"

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
def download_optimized(file_path: str, session_id: str = Header(...)):
    result = download_optimized_model(file_path)
    return JSONResponse(content=result)


@app.post("/headerTest")
def headerTest(session_id: str = Header(...)):
    return {"session_id": session_id}
