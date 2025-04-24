# save as main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from pinecone import Pinecone
import os
import json

# ✅ Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "store2"

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.index(PINECONE_INDEX)

# ✅ FastAPI app
app = FastAPI()

class Validation(BaseModel):
    field: str
    operator: str
    value: dict | list | str | int

class Rule(BaseModel):
    ruleId: str
    category: str
    subCategory: str
    ruleType: str
    description: str
    validation: Validation

@app.post("/embed-and-upsert")
async def embed_and_upsert(rules: List[Rule]):
    vectors = []
    for rule in rules:
        rule_dict = rule.dict()
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=json.dumps(rule_dict, ensure_ascii=False),
            task_type="RETRIEVAL_DOCUMENT"
        )["embedding"]

        vectors.append({
            "id": rule.ruleId,
            "values": embedding,
            "metadata": {
                "category": rule.category,
                "subCategory": rule.subCategory,
                "ruleType": rule.ruleType,
                "description": rule.description,
                "validation": json.dumps(rule.validation.dict())
            }
        })

    index.upsert(vectors=vectors)
    return {"status": "success", "count": len(vectors)}
