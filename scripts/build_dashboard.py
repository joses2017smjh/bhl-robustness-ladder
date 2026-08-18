"""Inject the current curves.json into the dashboard template -> docs/index.html."""
import json, datetime
from pathlib import Path

d = json.loads(Path("results/curves/curves.json").read_text())
d["stamp"] = "updated " + datetime.date.today().isoformat()
tpl = Path("scripts/dashboard_template.html").read_text()
Path("docs/index.html").write_text(tpl.replace("__DATA__", json.dumps(d, separators=(",", ":"))))
print(f"docs/index.html  {Path('docs/index.html').stat().st_size//1024} KB")
