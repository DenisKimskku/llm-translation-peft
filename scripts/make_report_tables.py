"""Collect outputs/*/score.json into a markdown + html results table. python scripts/make_report_tables.py [names...]"""
import json, glob, os, sys
names = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("outputs/*/score.json"))
cols = ["single_en", "single_ko", "mixed_en", "mixed_ko", "single", "mixed", "score"]
rows = []
for n in names:
    p = f"outputs/{n}/score.json"
    if not os.path.exists(p): continue
    s = json.load(open(p)); meta = {}
    mp = f"outputs/{n}/adapter/train_meta.json"
    if os.path.exists(mp): meta = json.load(open(mp))
    rows.append((n, meta.get("trainable_params"), meta.get("n_train"), [s.get(c) for c in cols]))
print("| run | params | n_train | " + " | ".join(cols) + " |"); print("|" + "---|" * (len(cols) + 3))
for n, tp, nt, v in rows:
    print(f"| {n} | {tp:,} | {nt:,} | " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in v) + " |" if tp else f"| {n} | - | - | " + " | ".join(f"{x:.2f}" for x in v) + " |")
html = "<table><tr><th>run</th><th>params</th><th>n_train</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
for n, tp, nt, v in rows:
    html += f"<tr><td>{n}</td><td>{tp:,}</td><td>{nt:,}</td>" if tp else f"<tr><td>{n}</td><td>-</td><td>-</td>"
    html += "".join(f"<td>{x:.2f}</td>" for x in v) + "</tr>"
html += "</table>"
open("report/_results_table.html", "w").write(html)
