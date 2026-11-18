# CNN-FineTuning-FC
FC 레이어만 파인 튜닝

## 로컬 세팅 (테스트용)

로컬에서는 코드가 끝까지 도는지만 확인하고, 실제 학습은 Kaggle에서 한다.
데이터셋은 받지 않는다. 테스트가 가짜 이미지를 만들어 쓴다.

torch는 CPU 빌드를 먼저 따로 설치한다 (기본 PyPI의 리눅스용 torch는 CUDA 포함이라 크다).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest tests -q
```
