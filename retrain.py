import json
from pathlib import Path
import joblib, mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

INFO_PATH = Path("models/model_info.json")

def _score(model, df):
    return f1_score(df["label"], model.predict(df["text"]), average="macro")

def retrain_if_needed(new_df, train_df, threshold=0.85, dry_run=False):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sentimen-video-games")

    info = json.loads(INFO_PATH.read_text())
    active = info["active_version"]
    old_model = joblib.load(info["models"][active]["file"])

    # bagi data baru: 80% buat belajar lagi, 20% buat uji adil lama vs baru
    cut = int(len(new_df) * 0.8)
    new_learn, new_test = new_df.iloc[:cut], new_df.iloc[cut:]
    old_score = _score(old_model, new_test)

    with mlflow.start_run(run_name=f"monitoring_{active}"):
        mlflow.log_metric("macro_f1_data_baru", old_score)
        mlflow.set_tag("stage", "monitoring")

    print(f"Model aktif {active} | macro-F1 di data baru = {old_score:.4f} | batas = {threshold}")
    if old_score >= threshold:
        print("Performa aman, tidak perlu re-training.")
        return {"retrained": False, "score": round(old_score, 4)}

    print("Performa di bawah batas -> perlu re-training.")
    if dry_run:
        print("(dry_run: re-training tidak dijalankan)")
        return {"retrained": "dry_run", "score": round(old_score, 4)}

    new_version = f"v{max(int(v[1:]) for v in info['models']) + 1}"
    data = pd.concat([train_df, new_learn])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                  max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=500)),
    ])
    model.fit(data["text"], data["label"])
    new_score = _score(model, new_test)

    file = f"models/model_{new_version}.pkl"
    joblib.dump(model, file)
    with mlflow.start_run(run_name=f"{new_version}_retrain"):
        mlflow.log_metrics({"macro_f1": new_score, "macro_f1_model_lama": old_score})
        mlflow.set_tag("version", new_version)
        mlflow.log_artifact(file)

    info["models"][new_version] = {"file": file, "macro_f1": round(new_score, 4)}
    if new_score > old_score:
        info["active_version"] = new_version
        print(f"{new_version} lebih bagus ({new_score:.4f}) -> jadi model aktif.")
    else:
        print(f"{new_version} tidak lebih bagus ({new_score:.4f}) -> model lama tetap dipakai.")
    INFO_PATH.write_text(json.dumps(info, indent=2))
    return {"retrained": True, "version": new_version, "score": round(new_score, 4)}
