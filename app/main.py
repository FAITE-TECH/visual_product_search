from fastapi import FastAPI
from app.database import Base, engine
from app.routers.products import router as products_router
from app.routers.visual_search import router as visual_search_router
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Visual Product Search API")

app.include_router(products_router, prefix="/api")
app.include_router(visual_search_router, prefix="/api")
