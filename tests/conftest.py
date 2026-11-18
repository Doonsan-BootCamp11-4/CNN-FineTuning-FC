"""테스트용 가짜 데이터셋: 3클래스 × 4장(train, train, val, test) = 12장.

conftest.py는 pytest가 자동으로 읽는 파일이다. 여기 정의한 fixture는
tests/ 안의 모든 테스트에서 import 없이 인자 이름(config)만 적으면 받아 쓸 수 있다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from configs.base import BASE


@pytest.fixture
def config(tmp_path: Path) -> dict:
    # tmp_path: pytest가 테스트마다 새로 만들어 주는 임시 폴더.
    # 테스트끼리 파일(이미지, best.pt 등)을 공유하지 않으므로 서로 영향을 주지 않는다.

    # 1) 가짜 이미지 만들기: <tmp_path>/<클래스>/<번호>.jpg
    rows = []  # split CSV에 쓸 행 목록
    for label in ("glass", "paper", "plastic"):
        (tmp_path / label).mkdir()
        # 클래스마다 4장: 0,1번은 train / 2번은 val / 3번은 test
        for i, split in enumerate(("train", "train", "val", "test")):
            path = f"{label}/{i}.jpg"  # data_root 기준 상대 경로 (실제 CSV와 같은 형식)
            # paper/0.jpg는 흑백 이미지 (실제 데이터셋에 RGB가 아닌 이미지가 섞여 있을 수 있음)
            mode = "L" if path == "paper/0.jpg" else "RGB"
            Image.new(mode, (40, 30), 100).save(tmp_path / path)  # 40×30 단색 이미지
            rows.append({"path": path, "label": label, "split": split})

    # 2) split CSV 쓰기: 실제 split CSV와 같은 컬럼(path, label, split)
    csv_path = tmp_path / "split.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # 3) BASE를 복사하고 테스트에 맞게 덮어쓴 config를 돌려준다
    return {
        **BASE,
        "name": "smoke",  # 결과 폴더 이름: <out_dir>/smoke/
        "epochs": 1,  # 끝까지 도는지만 보면 되므로 1번만
        "batch_size": 2,  # 이미지가 적으니 작게
        "num_workers": 0,  # Windows에서 worker 프로세스 문제 회피 (메인 프로세스에서 읽음)
        "num_classes": 3,  # 가짜 데이터의 클래스 수
        "split_csv": str(csv_path),  # 위에서 만든 CSV
        "data_root": str(tmp_path),  # CSV의 path 기준 폴더
        "out_dir": str(tmp_path / "out"),  # 결과도 임시 폴더에 저장 (저장소의 outputs/를 더럽히지 않음)
    }
