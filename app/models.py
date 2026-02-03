from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from .database import Base

class VisualProduct(Base):
    __tablename__ = "visual_products"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer)
    name = Column(String)
    image_path = Column(String)
    embedding = Column(Vector(512))
    source = Column(String)
