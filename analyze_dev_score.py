import csv
import math
import statistics


def safe_float(val):
    try:
        if val is None or val == "":
            return None
        return float(val)
    except ValueError:
        return None


data = []
with open("batch_corrected_dev.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert numeric fields
        processed_row = {
            "filename": row["filename"],
            "lipid_fit": safe_float(row.get("lipid_fit")),
            "aqueous_fit": safe_float(row.get("aqueous_fit")),
            "dev_score": safe_float(row.get("dev_score")),
            "gt_lipid": safe_float(row.get("gt_lipid")),
            "gt_aqueous": safe_float(row.get("gt_aqueous")),
            "score": safe_float(row.get("score")),
            "corr": safe_float(row.get("corr")),
        }
        data.append(processed_row)

# Filter rows with dev_score
dev_rows = [r for r in data if r["dev_score"] is not None]

print(f"Total rows: {len(data)}")
print(f"Rows with dev_score: {len(dev_rows)}")

if not dev_rows:
    print("No valid deviation scores found.")
    exit()

dev_scores = [r["dev_score"] for r in dev_rows]

print("\n--- Deviation Score Statistics ---")
print(f"Count: {len(dev_scores)}")
print(f"Mean: {statistics.mean(dev_scores):.4f}")
try:
    print(f"Median: {statistics.median(dev_scores):.4f}")
except:
    pass
print(f"Min: {min(dev_scores):.4f}")
print(f"Max: {max(dev_scores):.4f}")
if len(dev_scores) > 1:
    print(f"Std Dev: {statistics.stdev(dev_scores):.4f}")

# Check Ground Truth correlations
gt_rows = [r for r in dev_rows if r["gt_lipid"] is not None]

if gt_rows:
    print(f"\nRows with both dev_score and Ground Truth: {len(gt_rows)}")

    lipid_errors = []
    aqueous_errors = []
    dev_scores_gt = []
    scores_gt = []
    corrs_gt = []

    for r in gt_rows:
        l_err = abs(r["lipid_fit"] - r["gt_lipid"])
        a_err = abs(r["aqueous_fit"] - r["gt_aqueous"])
        lipid_errors.append(l_err)
        aqueous_errors.append(a_err)
        dev_scores_gt.append(r["dev_score"])
        scores_gt.append(r["score"])
        corrs_gt.append(r["corr"])

    # Simple correlation implementation
    def correlation(x, y):
        n = len(x)
        if n < 2:
            return 0
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom = math.sqrt(
            sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)
        )
        if denom == 0:
            return 0
        return numerator / denom

    corr_dev_lipid_err = correlation(dev_scores_gt, lipid_errors)
    corr_dev_aqueous_err = correlation(dev_scores_gt, aqueous_errors)

    print("\n--- Correlations ---")
    print(f"Correlation (dev_score vs lipid_error): {corr_dev_lipid_err:.4f}")
    print(f"Correlation (dev_score vs aqueous_error): {corr_dev_aqueous_err:.4f}")

    # Also check typical score vs error
    corr_score_lipid_err = correlation(scores_gt, lipid_errors)
    print(f"Correlation (score vs lipid_error): {corr_score_lipid_err:.4f}")

    # Split into High/Low dev score
    median_dev = statistics.median(dev_scores_gt)
    low_dev_rows = [r for r in gt_rows if r["dev_score"] < median_dev]
    high_dev_rows = [r for r in gt_rows if r["dev_score"] >= median_dev]

    low_dev_lipid_err = statistics.mean(
        [abs(r["lipid_fit"] - r["gt_lipid"]) for r in low_dev_rows]
    )
    high_dev_lipid_err = statistics.mean(
        [abs(r["lipid_fit"] - r["gt_lipid"]) for r in high_dev_rows]
    )

    print(f"\n--- Performance Split (Median Split at {median_dev:.2f}) ---")
    print(f"Mean Lipid Error (Low Dev Score): {low_dev_lipid_err:.2f} nm")
    print(f"Mean Lipid Error (High Dev Score): {high_dev_lipid_err:.2f} nm")

else:
    print("No ground truth data available to calculate errors.")
