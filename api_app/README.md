# RLID-NET Cloud API

RLID-NET을 클라우드에서 사용할 수 있도록 제공하는 REST API 서비스입니다.

## 🚀 빠른 시작

### 로컬 개발 환경

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일 편집
```

3. **Docker Compose로 실행**
```bash
docker-compose up -d
```

4. **API 문서 확인**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 API 사용 예시

### 1. 작업 제출

```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "inp_file=@path/to/your/file.inp" \
  -F "episodes=150" \
  -F "max_steps=50" \
  -F "runoff_weight=0.7" \
  -F "cost_weight=0.3" \
  -F "output_format=json"
```

**응답:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "작업이 제출되었습니다",
  "estimated_time_minutes": 30,
  "created_at": "2025-01-15T10:00:00Z"
}
```

### 2. 작업 상태 조회

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"
```

### 3. 결과 조회

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/results"
```

### 4. 파일 다운로드

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/download/excel" -o result.xlsx
```

## 🏗️ 아키텍처

- **FastAPI**: 웹 프레임워크
- **Celery**: 비동기 작업 처리
- **Redis**: 메시지 브로커
- **PostgreSQL**: 작업 메타데이터 저장
- **S3**: 파일 저장소

## 📝 환경 변수

```env
DATABASE_URL=postgresql://user:password@localhost:5432/rlidnet
REDIS_URL=redis://localhost:6379/0
S3_BUCKET_NAME=rlidnet-files
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

## 🚢 AWS 배포

자세한 배포 가이드는 `docs/CLOUD_DEPLOYMENT_PLAN.md`를 참조하세요.


