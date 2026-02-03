from fastapi import APIRouter, UploadFile, File
import shutil, os
from app.ai.image_encoder import encode_image
from app.database import SessionLocal
from app.models import VisualProduct

router = APIRouter()

UPLOAD_DIR = "app/uploads/company_products"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/company/products/upload")
def upload_product(file: UploadFile = File(...), name: str = "", company_id: int = 1):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    embedding = encode_image(path)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    elif isinstance(embedding, list):
        embedding = [float(x) for x in embedding]
    else:
        raise ValueError("Encoder returned invalid embedding")

    db = SessionLocal()
    product = VisualProduct(
        company_id=company_id,
        name=name,
        image_path=path,
        embedding=embedding,
        source="company"
    )
    db.add(product)
    db.commit()
    db.close()

    return {"status": "success", "product": name}
