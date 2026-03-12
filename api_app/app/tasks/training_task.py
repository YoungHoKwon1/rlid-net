#!/usr/bin/env python3
"""
Celery Task for RLID-NET Training
"""

import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from celery import Celery
from app.services.job_service import JobService
from app.models.job import JobStatus

# Celery 앱 초기화
celery_app = Celery(
    'rlidnet_tasks',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

job_service = JobService()
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='rlidnet.training')
def run_rlidnet_training(
    self,
    job_id: str,
    inp_file_path: str,
    episodes: int,
    max_steps: int,
    runoff_weight: float,
    cost_weight: float,
    output_format: str
):
    """
    RLID-NET 학습 작업 실행
    
    Args:
        job_id: 작업 ID
        inp_file_path: INP 파일 경로
        episodes: 에피소드 수
        max_steps: 최대 스텝 수
        runoff_weight: 유출수 가중치
        cost_weight: 비용 가중치
        output_format: 출력 형식
    """
    try:
        # 작업 상태를 RUNNING으로 업데이트
        job_service.update_job_status(job_id, JobStatus.RUNNING)
        
        # 환경 변수 설정
        os.environ['RLID_MAX_STEPS'] = str(max_steps)
        os.environ['RLID_RUNOFF_WEIGHT'] = str(runoff_weight)
        os.environ['RLID_COST_WEIGHT'] = str(cost_weight)
        
        # RLID-NET 학습 실행
        # 실제 구현: main.py의 로직을 여기로 이동하거나 호출
        from main import run_training_session, analyze_input_data, generate_final_reports
        from src.utils.config import create_default_config
        from src.core.swmm_simulator import analyze_example_inp
        
        logger.info(f"[Job {job_id}] 학습 시작: {episodes} episodes, {max_steps} steps")
        
        # 입력 데이터 분석
        analysis = analyze_example_inp(inp_file_path)
        
        # 설정 생성
        config = create_default_config(logger)
        config.experiment.base_inp_file = inp_file_path
        config.rl.num_episodes = episodes
        config.rl.max_steps_per_episode = max_steps
        config.rl.reward_runoff_weight = runoff_weight
        config.rl.reward_cost_weight = cost_weight
        
        # 출력 디렉토리 설정
        output_dir = Path(f"results/job_{job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        config.experiment.output_dir = str(output_dir)
        
        # 학습 실행
        agent, training_metrics, evaluation_results, visualizer, env = run_training_session(
            config, analysis, logger, inp_file_path
        )
        
        # 결과 생성
        generate_final_reports(
            agent, training_metrics, evaluation_results, visualizer, env, logger
        )
        
        # 결과 파일을 S3에 업로드
        result_files = upload_results_to_s3(job_id, output_dir)
        
        # 작업 결과 저장
        results = {
            "lid_placements": extract_lid_placements(env),
            "summary": extract_summary(training_metrics, env),
            "files": result_files
        }
        
        job_service.save_job_results(job_id, results)
        job_service.update_job_status(job_id, JobStatus.COMPLETED)
        
        logger.info(f"[Job {job_id}] 학습 완료")
        
        return {"status": "completed", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"[Job {job_id}] 학습 실패: {str(e)}", exc_info=True)
        job_service.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e)
        )
        raise

def upload_results_to_s3(job_id: str, output_dir: Path) -> dict:
    """결과 파일을 S3에 업로드"""
    import boto3
    s3_client = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET_NAME', 'rlidnet-files')
    
    files = {}
    
    # Excel 보고서
    excel_file = output_dir / "lid_placement_summary.xlsx"
    if excel_file.exists():
        s3_key = f"results/{job_id}/lid_placement_summary.xlsx"
        s3_client.upload_file(str(excel_file), bucket, s3_key)
        files["excel_report"] = f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    
    # 시각화 이미지
    viz_file = output_dir / "baseline_comparison.png"
    if viz_file.exists():
        s3_key = f"results/{job_id}/baseline_comparison.png"
        s3_client.upload_file(str(viz_file), bucket, s3_key)
        files["visualization"] = f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    
    # 학습 메트릭
    metrics_file = output_dir / "training_metrics.xlsx"
    if metrics_file.exists():
        s3_key = f"results/{job_id}/training_metrics.xlsx"
        s3_client.upload_file(str(metrics_file), bucket, s3_key)
        files["training_metrics"] = f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    
    return files

def extract_lid_placements(env):
    """LID 배치 정보 추출"""
    lid_summary = env.lid_manager.get_current_state_summary()
    return lid_summary.get('placements', [])

def extract_summary(training_metrics, env):
    """결과 요약 정보 추출"""
    final_reduction = training_metrics.episode_runoff_reductions[-1] if training_metrics.episode_runoff_reductions else 0
    final_cost = training_metrics.episode_costs[-1] if training_metrics.episode_costs else 0
    
    lid_summary = env.lid_manager.get_current_state_summary()
    total_area = sum(p['area_m2'] for p in lid_summary.get('placements', []))
    
    baseline_runoff = env.baseline_runoff
    reduction_rate = (final_reduction / baseline_runoff * 100) if baseline_runoff > 0 else 0
    
    return {
        "total_lid_area_m2": total_area,
        "total_cost_krw": final_cost,
        "runoff_reduction_m3": final_reduction,
        "runoff_reduction_percentage": reduction_rate,
        "cost_efficiency": final_reduction / (final_cost / 1000000) if final_cost > 0 else 0
    }


