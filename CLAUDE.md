# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 목적

사전학습된 ResNet18의 백본(합성곱 층)을 동결하고 FC(분류기) 레이어만 학습해 Garbage Classification(12클래스)을 푼다.
설계는 `docs/superpowers/specs/2026-11-10-fc-transfer-learning-design.md`.

## 구조

- `configs/base.py`: 공통 config `BASE` (경로, `num_workers`, 모델, `epochs`/`batch_size`/`seed`). 실험은 `{**BASE, "name": ..., <바꿀 값>}`.
- `configs/lr.py`, `scheduler.py`, `transform.py`: 팀원 각자의 실험 (비어 있음).
- `fcft/model.py`: `build_model(name="resnet18", num_classes=12)` — 백본 동결, FC만 학습.
- `fcft/data.py`: `build_loaders(config, train_transform=None)` — split CSV → train/val/test DataLoader. val/test 전처리는 `EVAL_TRANSFORM`으로 고정.
- `fcft/train.py`: `run(config, optimizer=None, scheduler=None, train_transform=None)` — optimizer·scheduler는 객체가 아니라 객체를 만드는 함수로 받는다. `<out_dir>/<name>/best.pt`, `history.json`(epoch마다 갱신) 저장.
- `fcft/evaluate.py`: `evaluate(config)` — best.pt로 test 평가, `metrics.json`·`confusion_matrix.png`·`curves.png` 저장.

지켜야 할 것:
- optimizer / scheduler / train transform은 config에 넣지 않고 `run`의 인자로 넘긴다.
- 학습 중에도 동결된 BatchNorm은 eval 모드로 둔다 (`freeze_bn_stats`). 이게 빠지면 running mean/var가 바뀐다.

## 실행

- 로컬은 테스트 전용(CPU, 데이터셋 없음), 실제 학습은 Kaggle GPU 노트북. 노트북 셀 순서는 README.
- venv 세팅: VS Code `Tasks: Run Task` → `venv: setup`, 또는 README의 명령.
- 테스트: `.venv\Scripts\python -m pytest tests -q` (tests/conftest.py가 임시 폴더에 가짜 이미지 12장과 split CSV를 만든다)
