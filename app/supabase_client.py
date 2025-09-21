import os
from supabase import create_client, Client
from io import BytesIO

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_report(description, image_file, embedding):
    # Upload image to Supabase Storage
    image_file.seek(0)
    image_data = image_file.read()
    file_name = f"{description[:10]}_{np.random.randint(1000)}.jpg"
    supabase.storage.from_("reports").upload(file_name, image_data)

    # Insert report record
    supabase.table("reports").insert({
        "description": description,
        "embedding": embedding.tolist(),
        "image_path": file_name
    }).execute()

    return f"https://{SUPABASE_URL}/storage/v1/object/public/reports/{file_name}"

def check_duplicate(new_embedding, threshold=0.85):
    # Fetch all embeddings from Supabase
    records = supabase.table("reports").select("embedding").execute().data

    if not records:
        return False

    new_emb = np.array(new_embedding)
    for r in records:
        existing_emb = np.array(r['embedding'])
        similarity = np.dot(new_emb, existing_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(existing_emb))
        if similarity >= threshold:
            return True
    return False
