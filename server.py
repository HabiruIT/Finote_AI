# server.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

import inference_multi   # <--- IMPORT MODULE XỬ LÝ

app = FastAPI(title="NLP Transaction API")

class InferenceRequest(BaseModel):
    text: str
    reference_time: str | None = None

@app.post("/inference")
async def run_inference(req: InferenceRequest):
    # Convert reference time nếu có
    reference_dt = None
    if req.reference_time:
        try:
            reference_dt = datetime.strptime(req.reference_time, "%Y-%m-%d %H:%M:%S")
        except:
            reference_dt = None

    # GỌI inference_multi XỬ LÝ
    result = inference_multi.process(req.text, reference_dt)

    # TRẢ JSON Y CHANG inference_multi
    return result

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)