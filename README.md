# Travel Stress Score (TSS) — Notre Dame Women’s Basketball

## Executive Summary
This project builds a **Travel Stress Score (TSS)** to quantify the logistical and competitive demands of travel for Notre Dame Women’s Basketball and assess how travel stress relates to on-court performance (scoring margin).

We produce:
- **Unweighted TSS (baseline):** a transparent index aggregating standardized travel and scheduling stressors plus interaction effects.
- **Weighted TSS\*** (data-derived): a tuned index where primary factors and interaction terms are weighted using historical outcomes via regression.

---

## Key Takeaways
- Travel stress is **multidimensional** (travel time, rest, opponent strength, time zones, home/away context, back-to-backs).
- Interaction terms capture **compounding stress** (e.g., long travel × time-zone shift).
- Data-derived weighting improves alignment with performance variability versus equal-weight baselines.
- The framework supports evidence-based scheduling and recovery planning.

---

## Full Methodology + Code
👉 **[`TSS_Methodology.qmd`](TSS_Methodology.qmd)**

---

## Repository Contents
- `README.md` — GitHub-rendered executive overview  
- `TSS_Methodology.qmd` — Full Quarto methodology + code  
- `2014-24 Final Data.xlsx` — Cleaned dataset  
- `ND WBB TSS Presentation.pdf` — Final presentation  
