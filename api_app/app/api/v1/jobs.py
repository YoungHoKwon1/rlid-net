#!/usr/bin/env python3
"""
Jobs API Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
import uuid
from datetime import datetime

from app.models.request import JobSubmitRequest, OutputFormat
from app.models.job import JobResponse, JobResultResponse, JobListResponse, JobStatus
from app.services.job_service import JobService
from app.services.rlidnet_service import RLIDNetService
from app.tasks.training_task import run_rlidnet_training

router = APIRouter()

job_service = JobService()
rlidnet_service = RLIDNetService()

@router.post("/jobs", response_model=JobResponse, status_code=202)
async def submit_job(
    background_tasks: BackgroundTasks,
    inp_file: UploadFile = File(..., description="SWMM INP 파일"),
    episodes: int = 150,
    max_steps: int = 50,
    runoff_weight: float = 0.7,
    cost_weight: float = 0.3,
    output_format: OutputFormat = OutputFormat.JSON
):
    """
    RLID-NET 학습 작업 제출
    
    - **inp_file**: SWMM 입력 파일 (.inp)
    - **episodes**: 학습 에피소드 수 (기본값: 150)
    - **max_steps**: 에피소드당 최대 스텝 수 (기본값: 50)
    - **runoff_weight**: 유출수 저감 가중치 (기본값: 0.7)
    - **cost_weight**: 비용 가중치 (기본값: 0.3)
    - **output_format**: 출력 형식 (json, excel, all)
    """
    # 입력 검증
    if not inp_file.filename.endswith('.inp'):
        raise HTTPException(status_code=400, detail="INP 파일만 업로드 가능합니다")
    
    if abs(runoff_weight + cost_weight - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail="runoff_weight와 cost_weight의 합은 1.0이어야 합니다"
        )
    
    # 작업 ID 생성
    job_id = str(uuid.uuid4())
    
    # 파일 저장 및 검증
    try:
        file_path = await rlidnet_service.save_uploaded_file(inp_file, job_id)
        await rlidnet_service.validate_inp_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"INP 파일 검증 실패: {str(e)}")
    
    # 작업 생성
    job_data = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "inp_file_path": file_path,
        "episodes": episodes,
        "max_steps": max_steps,
        "runoff_weight": runoff_weight,
        "cost_weight": cost_weight,
        "output_format": output_format.value,
        "created_at": datetime.utcnow()
    }
    
    job_service.create_job(job_data)
    
    # 예상 소요 시간 계산 (대략적)
    estimated_minutes = int(episodes * max_steps * 0.1)  # 대략적인 계산
    
    # 백그라운드 작업 시작
    background_tasks.add_task(
        run_rlidnet_training,
        job_id=job_id,
        inp_file_path=file_path,
        episodes=episodes,
        max_steps=max_steps,
        runoff_weight=runoff_weight,
        cost_weight=cost_weight,
        output_format=output_format.value
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="작업이 제출되었습니다",
        estimated_time_minutes=estimated_minutes,
        created_at=job_data["created_at"]
    )

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    작업 상태 조회
    
    - **job_id**: 작업 ID
    """
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    return JobResponse(**job)

@router.get("/jobs/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """
    작업 결과 조회
    
    - **job_id**: 작업 ID
    """
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"작업이 아직 완료되지 않았습니다. 현재 상태: {job['status']}"
        )
    
    results = job_service.get_job_results(job_id)
    
    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        results=results.get("results"),
        files=results.get("files")
    )

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="상태 필터"),
    limit: int = Query(20, ge=1, le=100, description="페이지 크기"),
    offset: int = Query(0, ge=0, description="페이지 오프셋")
):
    """
    작업 목록 조회
    
    - **status**: 상태 필터 (선택사항)
    - **limit**: 페이지 크기 (기본값: 20)
    - **offset**: 페이지 오프셋 (기본값: 0)
    """
    jobs, total = job_service.list_jobs(status=status, limit=limit, offset=offset)
    
    return JobListResponse(
        jobs=[JobResponse(**job) for job in jobs],
        total=total,
        limit=limit,
        offset=offset
    )

@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    """
    작업 취소
    
    - **job_id**: 작업 ID
    """
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail="완료되었거나 실패한 작업은 취소할 수 없습니다"
        )
    
    job_service.cancel_job(job_id)
    
    return None


