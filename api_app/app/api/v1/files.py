#!/usr/bin/env python3
"""
Files API Endpoints
"""

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
import zipfile
import io

from app.services.job_service import JobService
from app.services.rlidnet_service import RLIDNetService

router = APIRouter()

job_service = JobService()
rlidnet_service = RLIDNetService()

@router.get("/jobs/{job_id}/download/{file_type}")
async def download_result_file(
    job_id: str,
    file_type: str = Path(..., description="파일 타입: excel, visualization, training_metrics, all")
):
    """
    결과 파일 다운로드
    
    - **job_id**: 작업 ID
    - **file_type**: 다운로드할 파일 타입
      - `excel`: LID 배치 Excel 보고서
      - `visualization`: 시각화 이미지
      - `training_metrics`: 학습 메트릭 Excel
      - `all`: 모든 파일 (ZIP)
    """
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="작업이 완료되지 않았습니다"
        )
    
    results = job_service.get_job_results(job_id)
    files = results.get("files", {})
    
    if file_type == "all":
        # 모든 파일을 ZIP으로 압축
        return await download_all_files(job_id, files)
    elif file_type == "excel":
        file_url = files.get("excel_report")
        filename = "lid_placement_summary.xlsx"
    elif file_type == "visualization":
        file_url = files.get("visualization")
        filename = "baseline_comparison.png"
    elif file_type == "training_metrics":
        file_url = files.get("training_metrics")
        filename = "training_metrics.xlsx"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 타입: {file_type}"
        )
    
    if not file_url:
        raise HTTPException(
            status_code=404,
            detail=f"{file_type} 파일을 찾을 수 없습니다"
        )
    
    # S3에서 파일 다운로드
    file_path = await rlidnet_service.download_file_from_s3(file_url, job_id, file_type)
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

async def download_all_files(job_id: str, files: dict):
    """모든 결과 파일을 ZIP으로 압축하여 반환"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_type, file_url in files.items():
            if file_url:
                try:
                    file_path = await rlidnet_service.download_file_from_s3(
                        file_url, job_id, file_type
                    )
                    zip_file.write(file_path, file_path.name)
                except Exception as e:
                    # 파일이 없어도 계속 진행
                    pass
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=rlidnet_results_{job_id}.zip"
        }
    )


