"""Check required submission files without loading or executing checkpoints."""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
required = ["README.md", "report.md", "prediction.md", "figures/learning_curves.png"]
for run in ["baseline", "wide_300", "wide_900"]:
    required += [f"videos/{run}.mp4", f"checkpoints/{run}.pt"]
missing = [x for x in required if not (root / x).is_file() or (root / x).stat().st_size == 0]
if missing:
    print("Missing required files:")
    for item in missing:
        print(f"  {item}")
    sys.exit(1)
lines = []
for relative in required:
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {relative}")
(root / "submission_sha256.txt").write_text("\n".join(lines) + "\n")
print("All required submission files are present and non-empty.")
print("Wrote submission_sha256.txt. Checkpoint contents were not executed.")
