#!/usr/bin/env python3
"""
RLID-NET Cloud API - FastAPI Application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
from datetime import datetime

from app.api.v1 import jobs, files
from app.db.database import init_db

app = FastAPI(
    title="RLID-NET API",
    description="Reinforcement Learning for LID Placement Optimization API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 초기화
@app.on_event("startup")
async def startup_event():
    init_db()

# 라우터 등록
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
app.include_router(files.router, prefix="/api/v1", tags=["files"])

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "RLID-NET API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


