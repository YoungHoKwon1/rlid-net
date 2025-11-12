#!/usr/bin/env python3
"""
RLID-NET Batch Training Script
여러 설정으로 연속 학습을 실행하는 스크립트
"""

import subprocess
import time
import os
import argparse
from datetime import datetime

def run_training(episodes, steps, runoff_weight=0.7, cost_weight=0.3, output_suffix="", inp_file="inp_file/Example1.inp"):
    """단일 학습 실행"""
    print(f"\n{'='*60}")
    print(f"학습 시작: {episodes} episodes, {steps} steps")
    print(f"가중치: runoff_weight={runoff_weight}, cost_weight={cost_weight}")
    print(f"입력 파일: {inp_file}")
    print(f"{'='*60}")
    
    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results/batch_{episodes}_{steps}_{timestamp}{output_suffix}"
    
    # 명령어 구성
    cmd = [
        'python', 'main.py',
        '--episodes', str(episodes),
        '--output-dir', output_dir,
        '--inp-file', inp_file
    ]
    
    # 환경변수로 설정 값들 전달
    env = os.environ.copy()
    env['RLID_MAX_STEPS'] = str(steps)
    env['RLID_RUNOFF_WEIGHT'] = str(runoff_weight)
    env['RLID_COST_WEIGHT'] = str(cost_weight)
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"Step 수: {steps}")
    print(f"Runoff 가중치: {runoff_weight}")
    print(f"Cost 가중치: {cost_weight}")
    
    start_time = time.time()
    
    try:
        # 실시간 출력으로 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # 실시간 출력
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        end_time = time.time()
        
        if process.returncode == 0:
            print(f"\n학습 완료! (소요시간: {end_time-start_time:.1f}초)")
            return True
        else:
            print(f"\n학습 실패 (종료 코드: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='RLID-NET Batch Training Script')
    parser.add_argument('--inp-file', type=str, default='inp_file/Example1.inp',
                       help='Input SWMM INP file for the batch runs')
    args = parser.parse_args()

    print("RLID-NET Batch Training 시작")
    print(f"입력 파일: {args.inp_file}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 실행할 설정들
    configurations = [

    # (800, 50, 0.90, 0.10, "w9010_seed0_8lids_newcosts_643216_seocho"),
    # (800, 50, 1.00, 0.00, "w1000_seed1_8lids_newcosts_643216_seocho"),
    (800, 50, 0.50, 0.50, "w5050_seed0_8lids_3.3737_643216_seocho2subc4_2022full"),    
    # (800, 50, 0.00, 1.00, "w0100_seed0_8lids_3.3737_643216_seocho_2022"),    
    # (800, 50, 0.00, 1.0, "w0100_seed0_8lids_newcosts_643216"),
    # (800, 50, 0.00, 1.0, "w0100_seed1_8lids_newcosts_643216"),
    # (800, 50, 0.00, 1.0, "w0100_seed2_8lids_newcosts_643216"),
    # (800, 50, 0.00, 1.0, "w0100_seed3_8lids_newcosts_643216"),

    # (800, 50, 0.10, 0.90, "w1090_seed0_8lids_newcosts_643216"),
    # (800, 50, 0.10, 0.90, "w1090_seed1_8lids_newcosts_643216"),
    
    # (800, 50, 0.20, 0.80, "w2080_seed0_8lids_newcosts_643216"),
    # (800, 50, 0.20, 0.80, "w2080_seed1_8lids_newcosts_643216"),
    # (800, 50, 0.20, 0.80, "w2080_seed2_8lids_newcosts_643216"),
    # (800, 50, 0.20, 0.80, "w2080_seed3_8lids_newcosts_643216"),
    # (800, 50, 0.20, 0.80, "w2080_seed4_8lids_newcosts_643216"),


    # (800, 50, 0.80, 0.20, "w8020_seed0_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed1_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed2_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed3_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed4_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed5_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed6_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed7_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed8_8lids_newcosts_643216"),
    # (800, 50, 0.80, 0.20, "w8020_seed9_8lids_newcosts_643216"),

    # (800, 50, 0.90, 0.10, "w9010_seed0_8lids_newcosts_643216"),
    # (800, 50, 0.90, 0.10, "w9010_seed1_8lids_newcosts_643216"),
    # (800, 50, 0.90, 0.10, "w9010_seed2_8lids_newcosts_643216"),
    # (800, 50, 0.90, 0.10, "w9010_seed3_8lids_newcosts_643216"),
    # (800, 50, 0.90, 0.10, "w9010_seed4_8lids_newcosts_643216"),

    # (800, 50, 0.50, 0.50, "w5050_seed0_8lids_newcosts_643216"),
    # (800, 50, 0.50, 0.50, "w5050_seed1_8lids_newcosts_643216"),
    # (800, 50, 0.50, 0.50, "w5050_seed2_8lids_newcosts_643216"),
    # (800, 50, 0.50, 0.50, "w5050_seed3_8lids_newcosts_643216"),
    # (800, 50, 0.50, 0.50, "w5050_seed4_8lids_newcosts_643216"),
    ]
    
    results = []
    
    for i, (episodes, steps, runoff_weight, cost_weight, suffix) in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] 설정 실행 중...")
        
        success = run_training(episodes, steps, runoff_weight, cost_weight, suffix, args.inp_file)
        results.append({
            'episodes': episodes,
            'steps': steps,
            'runoff_weight': runoff_weight,
            'cost_weight': cost_weight,
            'suffix': suffix,
            'success': success,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # 다음 실행 전 잠시 대기
        if i < len(configurations):
            print(f"\n10초 후 다음 설정 실행...")
            time.sleep(10)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("배치 학습 결과 요약")
    print(f"{'='*60}")
    
    successful = 0
    for result in results:
        status = "성공" if result['success'] else "실패"
        print(f"Episodes: {result['episodes']:3d}, Steps: {result['steps']:3d}, "
              f"W: {result['runoff_weight']:.1f}/{result['cost_weight']:.1f} "
              f"({result['suffix']}) - {status}")
        if result['success']:
            successful += 1
    
    print(f"\n총 {len(results)}개 설정 중 {successful}개 성공")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 