# Dataset provenance

`student_data.csv`: original project author's 20 synthetic demonstration rows. Preserved unchanged. Five numeric inputs, target Final_Marks (0–100 scale). No real student data claimed.

`student-mat.csv` (optional, installed by fetch_dataset.py): Paulo Cortez (2008), Student Performance, UCI Machine Learning Repository, https://doi.org/10.24432/C5TG7T. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Source: https://archive.ics.uci.edu/dataset/320/student+performance. The mathematics subject subset has 395 records; the project extracts selected numerical columns only. No real-data file is present in the initial package.

| Column | Meaning |
|---|---|
| G1 | First-period grade, 0–20 |
| G2 | Second-period grade, 0–20 |
| studytime | Weekly study category: 1 <2h, 2 2–5h, 3 5–10h, 4 >10h |
| absences | Number of absences |
| failures | Past class failures (encoded in source) |
| freetime | Free time after school, 1–5 |
| G3 | Final grade, 0–20 (target) |

The no-G1/G2 experiment excludes earlier grades, but is not proof of an early-year forecast because the timing of absence information is not established. No demographic or sensitive input columns are used by the models. The source CSV may include additional columns; they are not exposed by the dashboard.
