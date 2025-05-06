# TO DO -- Api end points:
# Add socket com for rt status updates
# Add protection against spam
# Add api key based access

# run wtih uvicorn --app-dir . TwoN2DEndPoint:app --reload

# @app.get("/test")
# def test():
#     return test_connection()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64

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


# @app.get("/test")
# def test():
#     return test_connection()

@app.get("/")
def root():
    return {"message": "2N2D API is running"}


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    contents = await file.read()
    base64_str = base64.b64encode(contents).decode('utf-8')
    result = load_onnx_model(base64_str)
    return JSONResponse(content=result)


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    base64_str = base64.b64encode(contents).decode('utf-8')
    result = load_csv_data(base64_str, file.filename)
    return JSONResponse(content=result)


@app.post("/optimize")
async def optimize(request: dict):
    input_features = request.get("input_features")
    target_feature = request.get("target_feature")
    epochs = request.get("max_epochs")

    def status_callback(status_update):
        print(f"STATUS: {status_update}")

    result = find_optimal_architecture(
        input_features=input_features,
        target_feature=target_feature,
        status_callback=status_callback,
        max_epochs=epochs
    )
    return JSONResponse(content=result)


@app.get("/download-optimized")
def download_optimized():
    from TwoN2D import current_model_path
    result = download_optimized_model(current_model_path)
    return JSONResponse(content=result)
