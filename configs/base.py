"""모두가 같이 쓰는 기본 config. 실험은 {**BASE, "name": ..., <바꿀 값>} 형태로 만든다.

optimizer / scheduler / train transform은 여기 넣지 않고 run()의 인자로 넘긴다.
"""

BASE = {
    "name": "base",  # 실험 이름. 결과가 <out_dir>/<name>/ 에 저장된다

    # 실행 환경 (Kaggle 경로가 기본값, 로컬에서는 덮어쓴다)
    "split_csv": "/kaggle/working/CNN-FineTuning-Dataset-Split/splits/garbage12_602020_seed42.csv",  # train/val/test 분할 CSV (path, label, split)
    "data_root": "/kaggle/input/garbage-classification/garbage_classification",  # CSV의 path가 기준으로 삼는 이미지 루트 폴더
    "out_dir": "outputs",  # 모델(best.pt), history.json, metrics.json, 그래프를 저장할 폴더
    "num_workers": 2,  # DataLoader가 이미지를 읽을 때 쓰는 프로세스 수

    # 모델
    "model": "resnet18",  # 백본 이름 (지금은 resnet18만 지원)
    "num_classes": 12,  # 분류할 클래스 수 = FC 출력 크기

    # 학습
    "epochs": 15,  # 전체 train 데이터를 몇 번 반복 학습할지
    "batch_size": 32,  # 한 번에 모델에 넣는 이미지 수
    "seed": 42,  # 난수 시드 (가중치 초기화, 셔플 순서 고정)
}
