from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
import shutil
from ingest import  get_pdf_chunks, get_txt_chunks, get_doc_chunks
from pydantic import BaseModel
from typing import Optional


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app"""
    # Startup
    print("🚀 Starting IARAG API...")
    print("✅ API startup complete")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down IARAG API...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/process_doc")
async def process_(
    file: Optional[UploadFile] = File(None),  
    content: Optional[str] = Form(None),       
    output_dir: Optional[str] = Form(None),    
    language: str = Form("en"),                
    type_name: str = Form(...)          
):
    
    temp_path = None
    #save file in cache
    if file is not None:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    if type_name == "note":
        return get_txt_chunks(content)
    elif type_name == "pdf":
        if temp_path:
            return get_pdf_chunks(temp_path, output_dir, language)
        else:
            return {"error": "No file uploaded for PDF"}
    elif type_name == "doc":
        if temp_path:
            return get_doc_chunks(temp_path, output_dir, language)
        else:
            return {"error": "No file uploaded for DOC"}
    else:
        return {"error": "Unsupported type_name"}

    # clean cache
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)