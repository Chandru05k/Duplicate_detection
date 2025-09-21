from fastapi import FastAPI, UploadFile, File, Form
from app.embeddings import get_text_embedding, get_image_embedding
from app.supabase_client import insert_report, check_duplicate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your frontend domain to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/submit-report")
async def submit_report(
    description: str = Form(...),
    image: UploadFile = File(...)
):
    # 1. Generate embeddings
    text_emb = get_text_embedding(description)
    image_emb = get_image_embedding(image.file)

    # Combine embeddings (optional: can store separately too)
    combined_emb = list(text_emb) + list(image_emb)

    # 2. Check for duplicates in Supabase
    is_duplicate = check_duplicate(combined_emb)

    if is_duplicate:
        return {"status": "duplicate", "message": "This report seems similar to an existing report."}

    # 3. Insert new report
    report_url = await insert_report(description, image.file, combined_emb)

    return {"status": "success", "report_url": report_url}
