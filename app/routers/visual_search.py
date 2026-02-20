from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil, os
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import VisualProduct
from app.ai.image_encoder import encode_image
from sqlalchemy import bindparam
from pgvector.sqlalchemy import Vector

router = APIRouter()

UPLOAD_DIR = "app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/visual-search")
def visual_search(file: UploadFile = File(...), company_id: int = 1):
    db: Session = SessionLocal()
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        embedding = encode_image(file_path)
        print("Raw embedding:", embedding, "Type:", type(embedding))
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        embedding_param = bindparam(
            "embedding",
            value=embedding,
            type_=Vector(512)
        )

        distance_expr = VisualProduct.embedding.op("<=>")(embedding_param).label("distance")

        print(type(embedding), len(embedding), type(embedding[0]))

        results = (
            db.query(
                VisualProduct.id,
                VisualProduct.name,
                VisualProduct.image_path,
                VisualProduct.source,
                VisualProduct.embedding.l2_distance(embedding).label("distance")
            )
            .filter(VisualProduct.company_id == company_id)
            .order_by(VisualProduct.embedding.l2_distance(embedding))
            .limit(5)
            .all()
        )
        response = [
            {
                "id": r.id,
                "name": r.name,
                "image_path": r.image_path,
                "source": r.source,
                "distance": r.distance
            }
            for r in results
        ]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")
    
    finally:
        db.close()
