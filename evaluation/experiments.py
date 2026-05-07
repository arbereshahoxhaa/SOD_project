import pandas as pd

results = []

def log_result(name, iou, f1):
    results.append({
        "Model": name,
        "IoU": iou,
        "F1-score": f1
    })

def save_results():
    df = pd.DataFrame(results)
    df.to_csv("outputs/experiment_results.csv", index=False)
    print(df)