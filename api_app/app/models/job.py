#!/usr/bin/env python3
"""
Job Models for RLID-NET API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProgressInfo(BaseModel):
    """진행 상황 정보"""
    current_episode: int = Field(description="현재 에피소드")
    total_episodes: int = Field(description="전체 에피소드 수")
    percentage: float = Field(description="진행률 (%)")

class LIDPlacement(BaseModel):
    """LID 배치 정보"""
    lid_type: str = Field(description="LID 타입")
    area_m2: float = Field(description="면적 (m²)")
    area_percentage: float = Field(description="면적 비율 (%)")
    cost_krw: float = Field(description="비용 (KRW)")

class SummaryInfo(BaseModel):
    """결과 요약 정보"""
    total_lid_area_m2: float = Field(description="총 LID 면적 (m²)")
    total_cost_krw: float = Field(description="총 비용 (KRW)")
    runoff_reduction_m3: float = Field(description="유출수 저감량 (m³)")
    runoff_reduction_percentage: float = Field(description="유출수 저감률 (%)")
    cost_efficiency: float = Field(description="비용 효율성 (m³/M KRW)")

class ResultFiles(BaseModel):
    """결과 파일 정보"""
    excel_report: Optional[str] = Field(None, description="Excel 보고서 URL")
    visualization: Optional[str] = Field(None, description="시각화 이미지 URL")
    training_metrics: Optional[str] = Field(None, description="학습 메트릭 Excel URL")

class JobResponse(BaseModel):
    """작업 응답 모델"""
    job_id: str = Field(description="작업 ID")
    status: JobStatus = Field(description="작업 상태")
    message: str = Field(description="메시지")
    estimated_time_minutes: Optional[int] = Field(None, description="예상 소요 시간 (분)")
    created_at: datetime = Field(description="생성 시간")
    started_at: Optional[datetime] = Field(None, description="시작 시간")
    completed_at: Optional[datetime] = Field(None, description="완료 시간")
    progress: Optional[ProgressInfo] = Field(None, description="진행 상황")
    error: Optional[str] = Field(None, description="에러 메시지")

class JobResultResponse(BaseModel):
    """작업 결과 응답 모델"""
    job_id: str = Field(description="작업 ID")
    status: JobStatus = Field(description="작업 상태")
    results: Optional[Dict[str, Any]] = Field(None, description="결과 데이터")
    files: Optional[ResultFiles] = Field(None, description="결과 파일")

class JobListResponse(BaseModel):
    """작업 목록 응답 모델"""
    jobs: List[JobResponse] = Field(description="작업 목록")
    total: int = Field(description="전체 작업 수")
    limit: int = Field(description="페이지 크기")
    offset: int = Field(description="페이지 오프셋")


