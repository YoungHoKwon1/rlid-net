#!/usr/bin/env python3
"""
RLID-NET Service - RLID-NET 통합 서비스
"""

import os
import io
import tempfile
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError

class RLIDNetService:
    """RLID-NET 통합 서비스"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = os.getenv('S3_BUCKET_NAME', 'rlidnet-files')
        self.temp_dir = Path(tempfile.gettempdir()) / "rlidnet"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(self, uploaded_file, job_id: str) -> str:
        """업로드된 파일 저장"""
        file_path = self.temp_dir / f"{job_id}_{uploaded_file.filename}"
        
        with open(file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        
        # S3에도 백업 저장
        s3_key = f"input/{job_id}/{uploaded_file.filename}"
        self.s3_client.upload_fileobj(
            io.BytesIO(content),
            self.s3_bucket,
            s3_key
        )
        
        return str(file_path)
    
    async def validate_inp_file(self, file_path: str) -> bool:
        """INP 파일 검증"""
        # 실제 구현: PySWMM으로 파일 검증
        # 예시: 파일이 존재하고 읽을 수 있는지 확인
        if not os.path.exists(file_path):
            raise ValueError("파일을 찾을 수 없습니다")
        
        # 간단한 검증: .inp 확장자 확인
        if not file_path.endswith('.inp'):
            raise ValueError("INP 파일이 아닙니다")
        
        return True
    
    async def download_file_from_s3(self, s3_url: str, job_id: str, file_type: str) -> Path:
        """S3에서 파일 다운로드"""
        # S3 URL에서 키 추출
        # 예: https://s3.amazonaws.com/bucket/key -> key
        s3_key = s3_url.split(f"{self.s3_bucket}/")[-1] if f"{self.s3_bucket}/" in s3_url else s3_url
        
        local_path = self.temp_dir / f"{job_id}_{file_type}"
        
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
        except ClientError as e:
            raise ValueError(f"S3 파일 다운로드 실패: {str(e)}")
        
        return local_path

