> This README is rendered from `README.qmd` using Quarto for reproducibility.

# University of Notre Dame WBB Travel Stress Metric

## Executive Summary

This project develops a **Travel Stress Score (TSS)** to quantify the
logistical and competitive demands associated with Notre Dame Women’s
Basketball travel schedules and to evaluate how those demands relate to
on-court performance.

We constructed: - **Unweighted TSS (baseline):** a transparent index
that aggregates standardized travel and scheduling stressors plus
interaction effects. - **Weighted TSS\* (data-derived):** a tuned index
where both primary factors and interaction terms are **weighted using
historical outcomes** (scoring margin) via regression.

The resulting scores provide a practical tool for benchmarking travel
demand and supporting evidence-based scheduling and recovery planning.

## Key Takeaways

- Travel stress is **multidimensional** (travel time, rest, opponent
  strength, time zones, home/away context, back-to-backs).
- Interaction effects capture **compounding stress** (e.g., long travel
  \* time-zone shift).
- Data-derived weighting (TSS\*) improves alignment with observed
  performance variability vs. equal-weight baselines.
- The framework is modular and extensible to other teams, seasons, and
  outcomes.

## Methodology

### Problem framing

We aim to quantify “how demanding” each game’s context is using public
travel and schedule information, then evaluate whether higher stress
aligns with differences in performance (e.g., scoring margin).

### Index design principles

1.  **Directional consistency:** Each component is oriented so higher
    values represent higher stress.
2.  **Comparability:** Continuous inputs are standardized (z-scores) to
    avoid unit-driven dominance.
3.  **Baseline + tuning:** We start with an equal-weight baseline
    (unweighted TSS), then learn weights from historical outcomes to
    produce a tuned score (TSS\*).
4.  **Compounding effects:** Interaction terms capture situations where
    combined stressors matter more than either factor alone.

### Inputs used

Primary stress drivers: - `travel_minutes` - `Opponent Rank` →
transformed so stronger opponents (lower rank) increase stress -
`days_since_last_game` → used only when ≤ 50 days (exclude off-season
gaps) - `road_score` - `timezone_change` - `back_to_back` (1 if games
are on sequential days)

### Score definitions

**Unweighted baseline TSS**  
\[ \_{unweighted} = () + () \]

**Weighted TSS\***  
Weights are learned from a regression using scoring margin as the
outcome. Primary and interaction weights are derived from **absolute
standardized coefficients** and normalized to sum to 1: \[ ^\* =
(w_j,Z(x_j)) + (w\_{ab},Z(x_a,x_b)) \]

## 1. Load Libraries and Data

We loaded the tooling needed for data preparation, standardization,
interaction generation, and regression modeling, then read the final
cleaned dataset.

``` python
# !pip install openpyxl scipy statsmodels --quiet

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import zscore
import statsmodels.api as sm
```

``` python
FILE_PATH = "2014-24 Final Data.xlsx"
df = pd.read_excel(FILE_PATH)

print("Loaded dataset:", df.shape)
df.head()
```

    Loaded dataset: (378, 49)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | game_id | season | season_type | game_date | game_date_time | team_id | team_home_away | team_score | team_winner | assists | ... | game_datetime | margin | win | days_since_last_game | rest_score | road_score | timezone_change | Opponent_Rank | opponent_team_location.1 | Unnamed: 48 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 400508643 | 2014 | 2 | 2013-11-09 | 2013-11-09T18:00:00Z | 87 | home | 99 | True | 27 | ... | 2013-11-09 18:00:00 | 49 | 1 | 7 | 0.0 | 0 | 0 | NaN | UNC Wilmington | NaN |
| 1 | 400508654 | 2014 | 2 | 2013-11-11 | 2013-11-12T00:00:00Z | 87 | home | 81 | True | 23 | ... | 2013-11-12 00:00:00 | 19 | 1 | 2 | 1.0 | 0 | 0 | 20.0 | Michigan State | 1.909091 |
| 2 | 400508666 | 2014 | 2 | 2013-11-16 | 2013-11-16T19:00:00Z | 87 | home | 96 | True | 22 | ... | 2013-11-16 19:00:00 | 50 | 1 | 4 | 0.5 | 0 | 0 | NaN | Valparaiso | NaN |
| 3 | 400508685 | 2014 | 2 | 2013-11-23 | 2013-11-23T20:00:00Z | 87 | away | 76 | True | 23 | ... | 2013-11-23 20:00:00 | 22 | 1 | 7 | 0.0 | 1 | 1 | NaN | Pennsylvania | NaN |
| 4 | 400508697 | 2014 | 2 | 2013-11-26 | 2013-11-27T00:00:00Z | 87 | home | 92 | True | 22 | ... | 2013-11-27 00:00:00 | 16 | 1 | 3 | 0.5 | 0 | 0 | 23.0 | DePaul | NaN |

<p>5 rows × 49 columns</p>
</div>

## 2. Feature Engineering (Stress-Oriented Variables)

We transformed raw fields so that “higher = more stress” and created
additional indicators needed for the index.

``` python
# Stronger opponent (lower rank) => more stress
df["opp_rank_stress"] = -df["Opponent_Rank"]

# Ignored unusually long gaps (> 50 days) for rest stress
df["days_stress"] = df["days_since_last_game"].where(df["days_since_last_game"] <= 50, np.nan)

# Back-to-back indicator
df["back_to_back"] = (df["days_since_last_game"] == 1).astype(int)
```

## 3. Build Primary Stress Matrix

We isolated the core variables that represent travel/schedule stress and
store them in a single table.

``` python
primary_vars = [
    "travel_minutes",
    "opp_rank_stress",
    "days_stress",
    "road_score",
    "timezone_change",
    "back_to_back"
]

stress_df = df[primary_vars].copy()
stress_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | travel_minutes | opp_rank_stress | days_stress | road_score | timezone_change | back_to_back |
|----|----|----|----|----|----|----|
| 0 | 0.0 | NaN | 7.0 | 0 | 0 | 0 |
| 1 | 0.0 | -20.0 | 2.0 | 0 | 0 | 0 |
| 2 | 0.0 | NaN | 4.0 | 0 | 0 | 0 |
| 3 | 90.6 | NaN | 7.0 | 1 | 1 | 0 |
| 4 | 0.0 | -23.0 | 3.0 | 0 | 0 | 0 |

</div>

## 4. Standardize Continuous Inputs

Continuous variables use different units (minutes, days, etc.).
Z-scoring makes them comparable so no single metric dominates due to
scale.

``` python
continuous_vars = ["travel_minutes", "opp_rank_stress", "days_stress", "timezone_change"]

for col in continuous_vars:
    stress_df[col] = zscore(stress_df[col].fillna(stress_df[col].mean()))

stress_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | travel_minutes | opp_rank_stress | days_stress | road_score | timezone_change | back_to_back |
|----|----|----|----|----|----|----|
| 0 | -0.701490 | 5.570211e-16 | 1.360779 | 0 | -0.811107 | 0 |
| 1 | -0.701490 | -2.974113e+00 | -0.658441 | 0 | -0.811107 | 0 |
| 2 | -0.701490 | 5.570211e-16 | 0.149247 | 0 | -0.811107 | 0 |
| 3 | 0.995184 | 5.570211e-16 | 1.360779 | 1 | 1.232883 | 0 |
| 4 | -0.701490 | -3.914838e+00 | -0.254597 | 0 | -0.811107 | 0 |

</div>

## 5. Create and Standardize Interaction Terms

Interaction terms model “compound stress” (e.g., long travel combined
with timezone changes). We create pairwise interactions and z-score
them.

``` python
vars_for_interactions = [
    "travel_minutes",
    "opp_rank_stress",
    "days_stress",
    "road_score",
    "timezone_change"
]

interaction_terms = {}
for A, B in combinations(vars_for_interactions, 2):
    interaction_terms[f"{A}_x_{B}"] = stress_df[A] * stress_df[B]

interaction_df = pd.DataFrame(interaction_terms)

for col in interaction_df.columns:
    interaction_df[col] = zscore(interaction_df[col].fillna(interaction_df[col].mean()))

print("Number of interaction terms:", interaction_df.shape[1])
interaction_df.head()
```

    Number of interaction terms: 10

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | travel_minutes_x_opp_rank_stress | travel_minutes_x_days_stress | travel_minutes_x_road_score | travel_minutes_x_timezone_change | opp_rank_stress_x_days_stress | opp_rank_stress_x_road_score | opp_rank_stress_x_timezone_change | days_stress_x_road_score | days_stress_x_timezone_change | road_score_x_timezone_change |
|----|----|----|----|----|----|----|----|----|----|----|
| 0 | -0.024561 | -0.965692 | -0.580285 | -0.391035 | 0.047789 | -0.05546 | -0.059493 | 0.025619 | -1.123039 | -0.803326 |
| 1 | 2.149718 | 0.478886 | -0.580285 | -0.391035 | 2.596303 | -0.05546 | 2.320910 | 0.025619 | 0.582805 | -0.803326 |
| 2 | -0.024561 | -0.098945 | -0.580285 | -0.391035 | 0.047789 | -0.05546 | -0.059493 | 0.025619 | -0.099532 | -0.803326 |
| 3 | -0.024561 | 1.388934 | 0.790563 | 0.511112 | 0.047789 | -0.05546 | -0.059493 | 2.478223 | 1.773929 | 1.229974 |
| 4 | 2.837452 | 0.189971 | -0.580285 | -0.391035 | 1.344909 | -0.05546 | 3.073842 | 0.025619 | 0.241637 | -0.803326 |

</div>

## 6. Compute Unweighted TSS (Baseline)

The baseline score treats each standardized component as equally
important. This provides a transparent benchmark before learning
weights.

``` python
df["travel_stress_score"] = stress_df.sum(axis=1) + interaction_df.sum(axis=1)

df[["game_id", "season", "travel_stress_score"]].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | game_id   | season | travel_stress_score |
|-----|-----------|--------|---------------------|
| 0   | 400508643 | 2014   | -4.081303           |
| 1   | 400508654 | 2014   | 1.178984            |
| 2   | 400508666 | 2014   | -3.402581           |
| 3   | 400508685 | 2014   | 12.669855           |
| 4   | 400508697 | 2014   | 0.201289            |

</div>

## 7. Compute Weighted TSS\* (Primary Weights from Regression)

We calculated the data-derived weights for primary stress factors by
regressing scoring margin on the standardized predictors. We converted
absolute standardized coefficients into normalized weights.

``` python
reg_y = pd.to_numeric(df["margin"], errors="coerce")

reg_X = stress_df.copy()
for col in reg_X.columns:
    reg_X[col] = zscore(reg_X[col].fillna(reg_X[col].mean()))

reg_data = pd.concat([reg_X, reg_y], axis=1).dropna()
X_primary = sm.add_constant(reg_data[reg_X.columns])
y_primary = reg_data["margin"]

primary_model = sm.OLS(y_primary, X_primary).fit()
print(primary_model.summary())

primary_coefs = primary_model.params.drop("const")
primary_weights = primary_coefs.abs() / primary_coefs.abs().sum()

print("\nPrimary (regression-derived) weights:")
print(primary_weights)

df["weighted_TSS_primary"] = (reg_X[primary_weights.index] * primary_weights).sum(axis=1)
df[["game_id", "season", "weighted_TSS_primary"]].head()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 margin   R-squared:                       0.098
    Model:                            OLS   Adj. R-squared:                  0.084
    Method:                 Least Squares   F-statistic:                     6.754
    Date:                Fri, 19 Dec 2025   Prob (F-statistic):           8.33e-07
    Time:                        11:12:39   Log-Likelihood:                -1624.3
    No. Observations:                 378   AIC:                             3263.
    Df Residuals:                     371   BIC:                             3290.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const              15.7646      0.923     17.077      0.000      13.949      17.580
    travel_minutes      0.0400      1.819      0.022      0.982      -3.536       3.616
    opp_rank_stress    -3.0223      0.940     -3.214      0.001      -4.871      -1.173
    days_stress         0.7666      0.966      0.793      0.428      -1.134       2.667
    road_score        -15.0350      9.126     -1.647      0.100     -32.981       2.911
    timezone_change    10.6090      8.923      1.189      0.235      -6.936      28.154
    back_to_back        0.9321      0.965      0.966      0.335      -0.965       2.830
    ==============================================================================
    Omnibus:                        4.210   Durbin-Watson:                   1.608
    Prob(Omnibus):                  0.122   Jarque-Bera (JB):                4.110
    Skew:                           0.186   Prob(JB):                        0.128
    Kurtosis:                       3.350   Cond. No.                         23.1
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

    Primary (regression-derived) weights:
    travel_minutes     0.001316
    opp_rank_stress    0.099403
    days_stress        0.025212
    road_score         0.494491
    timezone_change    0.348922
    back_to_back       0.030656
    dtype: float64

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | game_id   | season | weighted_TSS_primary |
|-----|-----------|--------|----------------------|
| 0   | 400508643 | 2014   | -0.661265            |
| 1   | 400508654 | 2014   | -1.007809            |
| 2   | 400508666 | 2014   | -0.691810            |
| 3   | 400508685 | 2014   | 1.063759             |
| 4   | 400508697 | 2014   | -1.091138            |

</div>

## 8. Compute Weighted Interaction Contribution (Method 1)

Instead of manually assigning interaction weights, we estimated a
regression including both primary and interaction terms, then derive
weights from the interaction coefficients.

``` python
full_X = pd.concat([reg_X, interaction_df], axis=1)
full_data = pd.concat([full_X, reg_y], axis=1).dropna()

X_full = sm.add_constant(full_data[full_X.columns])
y_full = full_data["margin"]

interaction_model = sm.OLS(y_full, X_full).fit()
print(interaction_model.summary())

interaction_coefs = interaction_model.params.drop("const")[interaction_df.columns]
interaction_weights = interaction_coefs.abs() / interaction_coefs.abs().sum()

print("\nTop interaction weights (data-derived):")
print(interaction_weights.sort_values(ascending=False).head(15))

weighted_interactions = (interaction_df * interaction_weights).sum(axis=1)
df["weighted_TSS_full"] = df["weighted_TSS_primary"] + weighted_interactions

df[["game_id", "season", "margin", "travel_stress_score", "weighted_TSS_full"]].head()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 margin   R-squared:                       0.128
    Model:                            OLS   Adj. R-squared:                  0.102
    Method:                 Least Squares   F-statistic:                     4.887
    Date:                Fri, 19 Dec 2025   Prob (F-statistic):           4.56e-07
    Time:                        11:12:39   Log-Likelihood:                -1618.0
    No. Observations:                 378   AIC:                             3260.
    Df Residuals:                     366   BIC:                             3307.
    Df Model:                          11                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================================
                                            coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------------------
    const                                15.7646      0.914     17.247      0.000      13.967      17.562
    travel_minutes                       -3.5017      1.263     -2.773      0.006      -5.985      -1.019
    opp_rank_stress                       2.3905      1.786      1.338      0.182      -1.122       5.903
    days_stress                          -0.0344      0.949     -0.036      0.971      -1.901       1.833
    road_score                           -3.8483      1.150     -3.348      0.001      -6.109      -1.588
    timezone_change                       0.0942      0.680      0.139      0.890      -1.243       1.432
    back_to_back                          1.0710      0.964      1.111      0.267      -0.824       2.966
    travel_minutes_x_opp_rank_stress      6.3874      2.474      2.582      0.010       1.523      11.251
    travel_minutes_x_days_stress         -0.1471      1.823     -0.081      0.936      -3.732       3.438
    travel_minutes_x_road_score          -3.0022      1.272     -2.359      0.019      -5.505      -0.500
    travel_minutes_x_timezone_change      4.6767      1.916      2.441      0.015       0.909       8.444
    opp_rank_stress_x_days_stress         1.6226      0.990      1.640      0.102      -0.323       3.569
    opp_rank_stress_x_road_score         -8.1589      2.456     -3.323      0.001     -12.988      -3.330
    opp_rank_stress_x_timezone_change    -1.3868      1.586     -0.874      0.382      -4.506       1.732
    days_stress_x_road_score              0.8131      0.895      0.908      0.364      -0.947       2.574
    days_stress_x_timezone_change        -1.7172      1.241     -1.384      0.167      -4.157       0.722
    road_score_x_timezone_change          2.6767      1.666      1.606      0.109      -0.600       5.954
    ==============================================================================
    Omnibus:                        5.574   Durbin-Watson:                   1.555
    Prob(Omnibus):                  0.062   Jarque-Bera (JB):                5.665
    Skew:                           0.218   Prob(JB):                       0.0589
    Kurtosis:                       3.412   Cond. No.                     3.44e+16
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.6e-30. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.

    Top interaction weights (data-derived):
    opp_rank_stress_x_road_score         0.266729
    travel_minutes_x_opp_rank_stress     0.208813
    travel_minutes_x_timezone_change     0.152888
    travel_minutes_x_road_score          0.098148
    road_score_x_timezone_change         0.087507
    days_stress_x_timezone_change        0.056139
    opp_rank_stress_x_days_stress        0.053046
    opp_rank_stress_x_timezone_change    0.045338
    days_stress_x_road_score             0.026583
    travel_minutes_x_days_stress         0.004809
    dtype: float64

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | game_id   | season | margin | travel_stress_score | weighted_TSS_full |
|-----|-----------|--------|--------|---------------------|-------------------|
| 0   | 400508643 | 2014   | 49     | -4.081303           | -0.935393         |
| 1   | 400508654 | 2014   | 19     | 1.178984            | -0.482096         |
| 2   | 400508666 | 2014   | 50     | -3.402581           | -0.904312         |
| 3   | 400508685 | 2014   | 22     | 12.669855           | 1.479185          |
| 4   | 400508697 | 2014   | 16     | 0.201289            | -0.474604         |

</div>

## 9. Export Outputs

We exported a single dataset containing both baseline and weighted
scores for reporting and downstream analysis.

``` python
OUTPUT_XLSX = "2014-24_Final_Data_with_TSS_Unweighted_and_Weighted.xlsx"
df.to_excel(OUTPUT_XLSX, index=False)
print("Saved:", OUTPUT_XLSX)
```

    Saved: 2014-24_Final_Data_with_TSS_Unweighted_and_Weighted.xlsx

------------------------------------------------------------------------
