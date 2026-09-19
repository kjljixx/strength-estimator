import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_path = "prediction_testing_data.jsonl"

print(f"Loading results from {input_path}")

rows = []

with open(input_path, "r", encoding="utf-8") as file:
  for line_number, line in enumerate(file, start=1):
    if not line.strip():
      continue

    data = json.loads(line)

    lines = ["Strength Estimator - Win Chains (8 ctx games/player)", "Strength Estimator - Elo Bins (8 ctx games/player)", "Raw Elo"]

    for bucket, (accuracy, count) in data["slices"]["elo_diff_bucket"].items():
      lower, upper = map(int, bucket.split("-"))

      rows.append({
        "elo": (lower + upper) / 2,
        "accuracy": accuracy,
        "count": count,
        "run": f"{lines[line_number-1]}",
      })

df = pd.DataFrame(rows)

print(
  f"Loaded {df['run'].nunique()} runs and "
  f"{len(df)} total bucket observations"
)

sns.lineplot(
  data=df,
  x="elo",
  y="accuracy",
  hue="run",
  legend=False,
)

sns.scatterplot(
  data=df,
  x="elo",
  y="accuracy",
  hue="run",
  size="count",
  sizes=(20, 250),
)

plt.xlabel("Elo Diff")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()