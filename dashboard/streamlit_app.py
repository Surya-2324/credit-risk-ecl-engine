"""Streamlit demo for the Lending Club credit risk ECL engine.

Run from the project folder (D:\\Credit_Risk):
    streamlit run dashboard/streamlit_app.py
"""
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit, logit

ASSETS = Path(__file__).parent / "assets"
FIGURES = ASSETS / "figures"
EXCLUDE_FROM_REASONS = {"addr_state"}   # Lesson 11: location never used as a reason

st.set_page_config(page_title="Credit Risk ECL Engine", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    with open(ASSETS / "bundle.pkl", "rb") as f:
        return pickle.load(f)


B = load_bundle()
MODEL = B["pd_model"]
FEATS = B["features"]
CAT = B["categorical"]
NUMERIC = [f for f in FEATS if f not in CAT]


# ---------------------------------------------------------------- model helpers
def calibrate(p):
    cal = B["calibrator"]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if cal["method"] == "intercept_shift":
        return expit(logit(p) + cal["intercept_shift"])
    return expit(cal["platt_slope"] * logit(p) + cal["platt_intercept"])


def predict_lgd(X):
    lg = B["lgd"]
    if lg["choice"] == "baseline":
        return np.full(len(X), lg["baseline"])
    if lg["choice"] == "two_stage":
        pf = lg["stage1"].predict_proba(X)[:, 1]
        return pf + (1 - pf) * lg["stage2"].predict(X)
    return lg["lgbm"].predict(X)


def build_frame(values):
    """One-row DataFrame with the exact columns and categories the models expect."""
    df = pd.DataFrame([{f: values.get(f, np.nan) for f in FEATS}])
    for f in NUMERIC:
        df[f] = df[f].astype(float)
    for c in CAT:
        df[c] = pd.Categorical(df[c], categories=B["categories"][c])
    return df


def describe(f, v):
    if pd.isna(v):
        return f"{f}: not provided"
    return B["reasons"][f].format(v=v)


def score(values, rate_pct):
    X = build_frame(values)
    pd_val = float(calibrate(MODEL.predict_proba(X)[:, 1])[0])

    contrib = MODEL.predict(X, pred_contrib=True)[0][:-1]
    shap = pd.Series(contrib, index=FEATS).drop(list(EXCLUDE_FROM_REASONS), errors="ignore")

    loan = values["loan_amnt"]
    r = rate_pct / 1200
    installment = loan * r / (1 - (1 + r) ** -36)
    interest = installment * 36 - loan
    lgd = float(np.clip(predict_lgd(X)[0], 0, 1))
    ead_frac = float(np.clip(B["ead"]["model"].predict(X)[0], 0, 1))
    loss_if_default = lgd * ead_frac * loan
    exp_profit = (1 - pd_val) * interest - pd_val * loss_if_default
    breakeven = interest / (interest + loss_if_default)

    row = X.iloc[0]
    risks = [describe(f, row[f]) for f, v in shap.sort_values(ascending=False).items() if v > 0][:4]
    strengths = [describe(f, row[f]) for f, v in shap.sort_values().items() if v < 0][:3]

    return {
        "decision": "APPROVE" if exp_profit > 0 else "DECLINE",
        "pd": pd_val, "breakeven": breakeven, "exp_profit": exp_profit,
        "interest": interest, "loss_if_default": loss_if_default,
        "lgd": lgd, "ead_frac": ead_frac, "installment": installment,
        "risks": risks, "strengths": strengths, "shap": shap, "row": row,
    }


def template_memo(res, values):
    lines = [
        f"**Recommendation:** {res['decision']}",
        "",
        f"**Risk summary:** Predicted probability of default is {res['pd']:.1%} against a "
        f"break-even of {res['breakeven']:.1%}. Expected profit is ${res['exp_profit']:,.0f}; "
        f"loss if default is ${res['loss_if_default']:,.0f} on a ${values['loan_amnt']:,.0f} loan.",
        "",
        "**Key risk factors:**",
        *[f"- {r}" for r in res["risks"] or ["None identified"]],
        "",
        "**Mitigating factors:**",
        *[f"- {s}" for s in res["strengths"] or ["None identified"]],
        "",
        "**Suggested next steps:** Analyst review of the factors above.",
    ]
    return "\n".join(lines)


def shap_chart(res):
    s = res["shap"]
    top = s.reindex(s.abs().sort_values(ascending=False).index[:8]).iloc[::-1]
    labels = [describe(f, res["row"][f]) for f in top.index]
    colors = ["#c0392b" if v > 0 else "#2e86c1" for v in top]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, top.values, color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Push on log-odds of default (red = riskier, blue = safer)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- form helpers
D = B["defaults"]


def num(f, label, step=1.0, lo=0.0, hi=None):
    return st.number_input(label, min_value=lo, max_value=hi, value=float(D[f]), step=step)


def pick(c, label):
    options = B["categories"][c]
    default = B["cat_defaults"][c]
    return st.selectbox(label, options, index=options.index(default) if default in options else 0)


# ---------------------------------------------------------------- page
st.title("Credit Risk ECL Engine")
st.caption("Demo built on public Lending Club data (36-month loans, 2007–2015). "
           "An educational portfolio project, not financial advice.")

tab_score, tab_portfolio, tab_perf, tab_card = st.tabs(
    ["Score an applicant", "Portfolio ECL", "Model performance", "Model card"])

with tab_score:
    with st.form("applicant"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Loan")
            loan = st.number_input("Loan amount ($)", min_value=1000.0, max_value=40000.0,
                                   value=10000.0, step=500.0)
            rate = st.number_input("Interest rate offered (%)", min_value=5.0, max_value=30.0,
                                   value=12.0, step=0.5)
            purpose = pick("purpose", "Purpose")
            home = pick("home_ownership", "Home ownership")
            verification = pick("verification_status", "Income verification")
            state = pick("addr_state", "State")
        with c2:
            st.subheader("Credit profile")
            fico = num("fico", "FICO score", step=5.0, lo=300.0, hi=850.0)
            income = num("annual_inc", "Annual income ($)", step=1000.0, lo=1000.0)
            dti = num("dti", "Debt-to-income ratio (%)", step=0.5)
            util = num("revol_util", "Credit card utilisation (%)", step=1.0)
            inq = num("inq_last_6mths", "Credit inquiries, last 6 months", step=1.0)
            history = num("credit_history_years", "Credit history (years)", step=0.5)

        with st.expander("More details"):
            e1, e2 = st.columns(2)
            with e1:
                emp = num("emp_years", "Employment length (years)", step=1.0, hi=10.0)
                revol_bal = num("revol_bal", "Revolving balance ($)", step=500.0)
                open_acc = num("open_acc", "Open credit lines", step=1.0)
                total_acc = num("total_acc", "Total credit lines", step=1.0)
            with e2:
                mort = num("mort_acc", "Mortgage accounts", step=1.0)
                delinq = num("delinq_2yrs", "Delinquencies, last 2 years", step=1.0)
                pub_rec = num("pub_rec", "Public records", step=1.0)
                bankr = num("pub_rec_bankruptcies", "Bankruptcies", step=1.0)

        submitted = st.form_submit_button("Score applicant")

    if submitted:
        values = {
            "loan_amnt": loan, "fico": fico, "annual_inc": income, "dti": dti,
            "revol_util": util, "inq_last_6mths": inq, "credit_history_years": history,
            "emp_years": emp, "revol_bal": revol_bal, "open_acc": open_acc,
            "total_acc": total_acc, "mort_acc": mort, "delinq_2yrs": delinq,
            "pub_rec": pub_rec, "pub_rec_bankruptcies": bankr,
            "loan_to_income": loan / income,
            "purpose": purpose, "home_ownership": home,
            "verification_status": verification, "addr_state": state,
        }
        res = score(values, rate)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Decision", res["decision"])
        m2.metric("Probability of default", f"{res['pd']:.1%}")
        m3.metric("Break-even PD", f"{res['breakeven']:.1%}")
        m4.metric("Expected profit", f"${res['exp_profit']:,.0f}")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Credit memo")
            st.markdown(template_memo(res, values))
        with right:
            st.subheader("Why: SHAP contributions")
            st.pyplot(shap_chart(res))

        st.caption(f"Monthly payment ${res['installment']:,.0f} · predicted LGD {res['lgd']:.0%} · "
                   f"predicted exposure at default {res['ead_frac']:.0%} of the loan. "
                   "Location is used by the model but never given as a reason.")

with tab_portfolio:
    st.subheader("2015 test portfolio: expected credit loss")
    summary = ASSETS / "ecl_summary.csv"
    if summary.exists():
        st.dataframe(pd.read_csv(summary), hide_index=True)
    if (FIGURES / "ecl_scenarios.png").exists():
        st.image(str(FIGURES / "ecl_scenarios.png"),
                 caption="Lifetime ECL under base, adverse and severe scenarios")

with tab_perf:
    charts = [
        ("roc.png", "ROC curve, 2014 validation"),
        ("ks.png", "KS chart: defaulters caught vs good borrowers lost"),
        ("calibration.png", "Calibration by decile"),
        ("fico_monotonic.png", "Monotonic constraint check on FICO"),
        ("shap_importance.png", "Global SHAP importance"),
        ("profit_curve.png", "Profit vs approval rate"),
        ("monitoring.png", "Quarterly monitoring: PSI and calibration ratio"),
    ]
    cols = st.columns(2)
    shown = 0
    for name, caption in charts:
        path = FIGURES / name
        if path.exists():
            cols[shown % 2].image(str(path), caption=caption)
            shown += 1
    if shown == 0:
        st.info("No charts found in dashboard/assets/figures. Run export_assets.py first.")

with tab_card:
    card = ASSETS / "model_card.md"
    st.markdown(card.read_text(encoding="utf-8") if card.exists() else "Model card not found.")
