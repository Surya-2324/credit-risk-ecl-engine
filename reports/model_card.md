# Model Card: Lending Club PD Model

## Model details
- **Type:** LightGBM gradient boosting with monotonic constraints, plus Platt/intercept calibration.
- **Champion/challenger:** WoE logistic-regression scorecard (validation AUC 0.645) as the explainable challenger.
- **Target:** default within the loan term (Charged Off / Default = 1, Fully Paid = 0).

## Intended use
- Estimating probability of default for 36-month unsecured personal loans at application.
- Inputs to approval decisions, expected-loss estimation, and pricing review.
- **Not intended for:** other loan products, terms, or markets without revalidation.

## Data
- Lending Club accepted loans, 36-month term, issued 2007–2015.
- Out-of-time split: train 2007–2013 (175,426 loans, 12.63% default), validation 2014 (162,570, 13.73%), test 2015 (283,026, 14.89%, held out).
- Unfinished loans excluded; vintages after 2015 excluded to avoid maturity bias.

## Features
- Application-time features only (whitelist): credit score, DTI, income, utilisation, inquiries, credit history, and others.
- **Excluded:** post-origination columns (leakage), and Lending Club grade / interest rate (another lender's model; used as a benchmark).
- Monotonic constraints on features with clear direction (e.g. higher FICO never increases PD).

## Performance (validation 2014)
- LightGBM AUC: 0.662 (Gini 0.323); Lending Club interest-rate benchmark: 0.658.
- Monotonic constraints cost about 0.001 AUC versus an unconstrained model.
- Without `addr_state`: AUC 0.661.

## Calibration
- Uncalibrated PDs under-predicted 2014 defaults by about 4% because of population drift.
- Class rebalancing (SMOTE-style) left AUC unchanged but inflated mean PD from 14% to 47%; not used.
- Calibration refreshed on the most recent year; monitored over time.

## Fairness
- No protected attributes available; geography tested as a proxy across US Census regions.
- Lowest Adverse Impact Ratio across regions: 0.97 (threshold for review: 0.80).
- Decision on `addr_state`: [WRITE YOUR DECISION AND REASON HERE].
- Reason codes exclude state of residence.
- A production deployment would add proxy-based testing (e.g. BISG).

## Limitations
- Only approved loans are observed (no reject inference), so risk among rejected applicants is unknown.
- Population drift: default rates rose from 12.6% to 14.9% across the split; calibration requires regular refresh.
- Profit analysis assumes full-term interest and ignores funding and operating costs.

## Monitoring plan
- Population Stability Index (PSI) on scores and key features; calibration traffic lights by decile; quarterly review.
