from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import base64
from TwoN2D import load_onnx_model, load_csv_data, find_optimal_architecture, download_optimized_model

app = FastAPI()

#TO DO -- Api end points:
#Add socket com for rt status updates
#Add protection against spam
#Add api key based acces

# @app.get("/test")
# def test():
#     return test_connection()

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    b64data = base64.b64encode(await file.read()).decode("utf-8")
    return load_onnx_model(b64data)

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    b64data = base64.b64encode(await file.read()).decode("utf-8")
    return load_csv_data(b64data, file.filename)

@app.post("/optimize")
async def optimize_model(input_features: list[str] = Form(...), target_feature: str = Form(...)):
    return find_optimal_architecture(input_features, target_feature)

@app.get("/download-optimized")
def download_model():
    result = download_optimized_model("/tmp/optimized_model.onnx")
    return JSONResponse(result)
