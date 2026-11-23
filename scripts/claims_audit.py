"""D3: claim audit of paper V2. Every number in the rewritten sections must come from a result file.

Usage: python scripts/claims_audit.py [--out docs/paper_v2/claims.csv]
For each sentence of the audited sections that contains a number, each number is looked up in (1) the generated
tables (docs/paper_v2/tables/*.tex), (2) every value of the result JSON files (results/paper_json), at the precisions
used in the text and also as a percentage, (3) a short list of design constants (sizes, counts of users, seeds).
Unmatched numbers are written with status CHECK; they must be traced by hand or removed.
"""
import argparse, csv, json, pathlib, re

SECTIONS = ["01_introduction", "05b_law", "06_results", "07_discussion", "08_conclusion"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
# Design constants of the protocol (not results): sizes, user counts, seeds, options, bins, years, section numbers.
CONST = {"0.8", "4", "9", "27", "30", "50", "10", "20", "99", "100", "2,000", "1,000", "10,000", "5,000", "8,000",
         "34,667", "3", "2", "1", "6", "7", "8", "5", "0.6", "0.3", "0.05", "0.01", "80", "95", "15", "14", "12",
         "16", "2026", "2027", "1,000", "0.15", "13", "0", "40", "200", "250", "128", "24", "19", "250"}


def json_numbers():
    """value string -> first (file, key path) holding it. Precision is kept as written, so 0.039 matches only values
    that round to 0.039; percentages match 100 x value."""
    vals = {}
    def walk(x, f, path):
        if isinstance(x, dict):
            for k, v in x.items(): walk(v, f, path + [str(k)])
        elif isinstance(x, list):
            for i, v in enumerate(x): walk(v, f, path + [str(i)])
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            for v in (x, 100 * x):
                for d in (0, 1, 2, 3, 4):
                    for t in (f"{v:.{d}f}", f"{-v:.{d}f}", f"{v:,.0f}"):
                        vals.setdefault(t.lstrip("-"), f"{f.name}:{'/'.join(path)}")
    for f in sorted((ROOT / "results/paper_json").rglob("*.json")):
        try:
            walk(json.loads(f.read_text()), f, [])
        except Exception:
            pass
    return vals


def table_numbers():
    vals = set()
    for f in (ROOT / "docs/paper_v2/tables").glob("*.tex"):
        vals |= set(re.findall(r"\d[\d,]*\.?\d*", f.read_text()))
    return vals


def clean(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(ref|label|cite[tp]?|eqref|input|url)\{[^}]*\}", " ", s)
    s = re.sub(r"\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/paper_v2/claims.csv"); a = ap.parse_args()
    J, T = json_numbers(), table_numbers()
    rows, n_check = [], 0
    texts = [("abstract", re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", (ROOT / "docs/paper_v2/main.tex").read_text(), re.S).group(1))]
    texts += [(s, (ROOT / f"docs/paper_v2/sections/{s}.tex").read_text()) for s in SECTIONS]
    for sec, txt in texts:
        for sent in re.split(r"(?<=[.;:])\s+(?=[A-Z\\])", clean(txt)):
            nums = re.findall(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*(?![\w])", sent.replace("$", " "))
            nums = [n.strip("+").rstrip(".,") for n in nums]
            nums = [n for n in nums if n and n not in CONST and not re.fullmatch(r"L?\d", n)]
            for n in nums:
                m = n.lstrip("-")
                src = "table" if m in T else J.get(m, "")
                # a number with fewer than three significant digits matched only in a JSON is weak evidence
                weak = src and src != "table" and len(re.sub(r"[^0-9]", "", m).lstrip("0")) < 3
                status = "OK" if src and not weak else ("REVIEW" if weak else "CHECK")
                n_check += status != "OK"
                rows.append([sec, n, src, status, " ".join(sent.split())[:220]])
    with open(ROOT / a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["section", "number", "source", "status", "sentence"]); w.writerows(rows)
    print(f"{len(rows)} numbers, {n_check} to check by hand -> {a.out}")
    for r in rows:
        if r[3] != "OK":
            print(f"  {r[3]:6s} {r[0]:14s} {r[1]:>8s} [{r[2][:45]}] | {r[4][:110]}")
