#!/usr/bin/env python3
"""
Request Models for RLID-NET API
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum

class OutputFormat(str, Enum):
    """출력 형식"""
    JSON = "json"
    EXCEL = "excel"
    ALL = "all"

class JobSubmitRequest(BaseModel):
    """작업 제출 요청 모델"""
    
    episodes: int = Field(
        default=150,
        ge=10,
        le=2000,
        description="학습 에피소드 수 (10-2000)"
    )
    
    max_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="에피소드당 최대 스텝 수 (10-200)"
    )
    
    runoff_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="유출수 저감 가중치 (0.0-1.0)"
    )
    
    cost_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="비용 가중치 (0.0-1.0)"
    )
    
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="출력 형식"
    )
    
    @validator('runoff_weight', 'cost_weight')
    def validate_weights(cls, v, values):
        """가중치 합이 1.0에 가까운지 검증"""
        if 'runoff_weight' in values and 'cost_weight' in values:
            total = values.get('runoff_weight', 0) + values.get('cost_weight', 0)
            if abs(total - 1.0) > 0.01:
                raise ValueError("runoff_weight와 cost_weight의 합은 1.0이어야 합니다")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "episodes": 150,
                "max_steps": 50,
                "runoff_weight": 0.7,
                "cost_weight": 0.3,
                "output_format": "json"
            }
        }


