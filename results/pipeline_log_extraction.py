import re
import pandas as pd

# 1) read log lines
log_path = "pipeline_log.txt"
with open(log_path, "r") as f:
    lines = [l.rstrip("\n") for l in f]

# 2) compile three regexes
agg_header = re.compile(r"\*\*\* AGGREGATED \[(?P<model>.+?)\] for '(?P<group>.+?)' \*\*\*")
metrics_re = re.compile(r"TP=(?P<tp>\d+), FP=(?P<fp>\d+), FN=(?P<fn>\d+), TN=(?P<tn>\d+)")
scores_re  = re.compile(r"Precision=(?P<precision>\d+\.\d+), Recall=(?P<recall>\d+\.\d+), F1=(?P<f1>\d+\.\d+), Accuracy=(?P<accuracy>\d+\.\d+)")

# 3) walk through lines and collect records using .search()
records = []
current = None
for line in lines:
    m = agg_header.search(line)
    if m:
        current = {"model": m.group("model"), "group": m.group("group")}
        continue
    if current:
        mm = metrics_re.search(line)
        if mm:
            current.update({k:int(v) for k,v in mm.groupdict().items()})
            continue
        ms = scores_re.search(line)
        if ms:
            current.update({k:float(v) for k,v in ms.groupdict().items()})
            records.append(current)
            current = None

# 4) turn into DataFrame
df_agg = pd.DataFrame(records)

# 5) sanity check and export
print(df_agg.head(5))

# df_agg.to_csv("aggregated_metrics.csv", index=False)