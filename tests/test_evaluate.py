from pathlib import Path

from fcft.evaluate import evaluate
from fcft.train import run


# [정상 동작] run()이 남긴 best.pt로 test를 평가하면
# - 정확도가 0~1 사이 값으로 나오고
# - confusion matrix가 클래스 수(3) × 클래스 수 크기이며, 칸의 합이 test 이미지 수(3)와 같고
# - metrics.json, confusion_matrix.png, curves.png 세 파일이 생기는지.
def test_evaluate(config: dict) -> None:
    out = Path(run(config)["out_dir"])
    metrics = evaluate(config)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    assert sum(map(sum, cm)) == 3
    for f in ("metrics.json", "confusion_matrix.png", "curves.png"):
        assert (out / f).exists()


# [출력 형식] per_class에는 클래스 이름만 키로 들어가고, 평균 지표는 따로 저장되는지.
# sklearn classification_report는 클래스 외에 accuracy, macro avg, weighted avg를 같은 dict에 섞어 돌려준다.
def test_per_class_only_classes(config: dict) -> None:
    run(config)
    metrics = evaluate(config)
    assert set(metrics["per_class"]) == {"glass", "paper", "plastic"}
    assert set(metrics["per_class"]["glass"]) >= {"precision", "recall", "f1-score"}
    assert "f1-score" in metrics["macro_avg"]
