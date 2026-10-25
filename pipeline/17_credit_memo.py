import os
import re
import json
import time
import pickle
import numpy as np
import pandas as pd
from scipy.special import logit, expit

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

def load(name):
    with open(f"data/processed/{name}.pkl", "rb") as f:
        return pickle.load(f)

lg_pd, cal, lg, ead = (load(n) for n in ["lgbm_model", "calibrator", "lgd_model", "ead_model"])
REASONS = load("reason_templates")
pd_model, FEATS, CATEGORICAL = lg_pd["model"], lg_pd["features"], lg_pd["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    test[c] = pd.Categorical(test[c].where(test[c].isin(cats)), categories=cats)

def calibrate(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if cal["method"] == "intercept_shift":
        return expit(logit(p) + cal["intercept_shift"])
    return expit(cal["platt_slope"] * logit(p) + cal["platt_intercept"])

def predict_lgd(X):
    if lg["choice"] == "baseline":
        return np.full(len(X), lg["baseline"])
    if lg["choice"] == "two_stage":
        pf = lg["stage1"].predict_proba(X)[:, 1]
        return pf + (1 - pf) * lg["stage2"].predict(X)
    return lg["lgbm"].predict(X)

EXCLUDE_FROM_REASONS = {"addr_state"}

def build_facts(row):
    X = row[FEATS]
    raw = pd_model.predict_proba(X)[:, 1]
    pdv = float(calibrate(raw)[0])

    contrib = pd_model.predict(X, pred_contrib=True)[0][:-1]
    s = pd.Series(contrib, index=FEATS).drop(list(EXCLUDE_FROM_REASONS), errors="ignore")

    def describe(f):
        v = X.iloc[0][f]
        return f"{f} not provided" if pd.isna(v) else REASONS[f].format(v=v)

    risks = [describe(f) for f, val in s.sort_values(ascending=False).items() if val > 0][:4]
    strengths = [describe(f) for f, val in s.sort_values().items() if val < 0][:3]

    loan = float(row["loan_amnt"].iloc[0])
    interest = float(row["installment"].iloc[0] * 36 - loan)
    lgd_v = float(np.clip(predict_lgd(X)[0], 0, 1))
    ead_frac = float(np.clip(ead["model"].predict(X)[0], 0, 1))
    loss_if_default = lgd_v * ead_frac * loan
    exp_profit = (1 - pdv) * interest - pdv * loss_if_default
    breakeven = interest / (interest + loss_if_default)

    return {
        "decision": "APPROVE" if exp_profit > 0 else "DECLINE",
        "loan_amount": f"${loan:,.0f}",
        "loan_purpose": str(X.iloc[0]["purpose"]),
        "probability_of_default": f"{pdv:.1%}",
        "break_even_pd": f"{breakeven:.1%}",
        "expected_profit": f"${exp_profit:,.0f}",
        "loss_if_default": f"${loss_if_default:,.0f}",
        "risk_factors": risks,
        "strengths": strengths,
    }

SYSTEM = """You are a credit analyst assistant at a consumer lender.
Write a short internal credit memo for a human credit analyst.

Rules:
- Use ONLY the facts in the JSON provided. Do not invent any numbers, facts, or borrower details.
- State the recommendation exactly as given in "decision". Do not change or second-guess it.
- Never mention race, ethnicity, religion, gender, age, nationality, marital status, or location.
- Plain, professional English. Under 180 words.

Use exactly these sections:
Recommendation:
Risk summary:
Key risk factors:
Mitigating factors:
Suggested next steps:"""

def call_llm(facts):
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    prompt = "Write the credit memo for this application.\n\nFACTS:\n" + json.dumps(facts, indent=2)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=500, temperature=0, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    if provider == "groq":
        from groq import Groq
        client = Groq()
        r = client.chat.completions.create(
            model=os.environ["LLM_MODEL"], temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content

    return None

NUM = re.compile(r"\d[\d,]*\.?\d*")
BANNED = [r"\brace\b", r"\bethnic", r"\breligio", r"\bgender\b", r"\bsex\b",
          r"\bage\b", r"\bnationality\b", r"\bmarital\b", r"\bstate of residence\b"]

def numbers(text):
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}

def check_memo(memo, facts):
    allowed = numbers(json.dumps(facts)) | {str(i) for i in range(1, 11)}
    invented = sorted(numbers(memo) - allowed)
    banned = [b for b in BANNED if re.search(b, memo.lower())]
    decision_ok = facts["decision"] in memo.upper()
    return {"invented_numbers": invented, "banned_terms": banned,
            "decision_matches": decision_ok,
            "passed": not invented and not banned and decision_ok}


def template_memo(f):
    lines = [
        f"Recommendation: {f['decision']}",
        f"Risk summary: Predicted probability of default is {f['probability_of_default']} "
        f"against a break-even of {f['break_even_pd']}. Expected profit is {f['expected_profit']}; "
        f"loss if default is {f['loss_if_default']}.",
        "Key risk factors:",
        *[f"- {r}" for r in f["risk_factors"] or ["None identified"]],
        "Mitigating factors:",
        *[f"- {s}" for s in f["strengths"] or ["None identified"]],
        "Suggested next steps: Analyst review of the factors above.",
    ]
    return "\n".join(lines)

def generate_memo(facts):
    start = time.time()
    try:
        memo = call_llm(facts)
    except Exception as e:
        print(f"  LLM error: {e}")
        memo = None
    seconds = time.time() - start

    if memo is None:
        return template_memo(facts), {"source": "template (no LLM)", "passed": True, "seconds": seconds}

    check = check_memo(memo, facts)
    if not check["passed"]:
        return template_memo(facts), {**check, "source": "template (LLM failed checks)", "seconds": seconds}
    return memo, {**check, "source": "llm", "seconds": seconds}


sample = test.sample(2000, random_state=42)
pds = calibrate(pd_model.predict_proba(sample[FEATS])[:, 1])
risky = sample.iloc[[int(np.argmax(pds))]]
safe = sample.iloc[[int(np.argmin(pds))]]

for label, row in [("Highest-risk applicant", risky), ("Lowest-risk applicant", safe)]:
    facts = build_facts(row)
    memo, meta = generate_memo(facts)
    print(f"\n{'=' * 60}\n{label}  [{meta['source']}, {meta['seconds']:.1f}s]\n{'=' * 60}")
    print(memo)


records, memos_md = [], []
for i, idx in enumerate(sample.index[:20]):
    facts = build_facts(sample.loc[[idx]])
    memo, meta = generate_memo(facts)
    records.append({"applicant": i, "decision": facts["decision"], **meta})
    memos_md.append(f"## Applicant {i} – {facts['decision']} ({meta['source']})\n\n{memo}\n")

ev = pd.DataFrame(records)
print("\nMemo sources:")
print(ev["source"].value_counts())
if (ev["source"] != "template (no LLM)").any():
    llm_rows = ev[ev["source"] != "template (no LLM)"]
    print(f"LLM pass rate: {(llm_rows['source'] == 'llm').mean():.0%}")
    print(f"Average latency: {llm_rows['seconds'].mean():.1f}s")

os.makedirs("reports", exist_ok=True)
ev.to_csv("reports/memo_evaluation.csv", index=False)
with open("reports/credit_memos.md", "w", encoding="utf-8") as f:
    f.write("# Sample credit memos\n\n" + "\n".join(memos_md))
print("\nSaved reports/credit_memos.md and reports/memo_evaluation.csv")