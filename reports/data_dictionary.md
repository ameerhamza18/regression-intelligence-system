# Data Dictionary: Rossmann Store Sales

Source: Kaggle "Rossmann Store Sales" (`train.csv` + `store.csv`), merged on `store`.
Columns are renamed to snake_case during ingestion.

| Column | Meaning | Role |
|---|---|---|
| store | Unique store ID | Key / feature |
| day_of_week | 1 = Monday … 7 = Sunday | Feature (categorical) |
| date | Calendar date | Key / source of calendar features |
| sales | Daily turnover | **Target** |
| customers | Customers that day | **Excluded (leakage)** |
| open | 0 = closed, 1 = open | Filter (drop closed days) |
| promo | Store runs a promotion that day | Feature |
| state_holiday | a = public, b = Easter, c = Christmas, 0 = none | Feature |
| school_holiday | Store affected by school closures | Feature |
| store_type | Store model: a, b, c, d | Feature |
| assortment | a = basic, b = extra, c = extended | Feature |
| competition_distance | Metres to nearest competitor | Feature |
| competition_open_since_month/year | When the nearest competitor opened | Feature (engineered) |
| promo2 | Store participates in continuing promotion | Feature |
| promo2_since_week/year | When the store joined Promo2 | Feature (engineered) |
| promo_interval | Months when Promo2 restarts | Feature (engineered) |

Known quirks: closed days have zero sales; some store attributes are missing by design
(Promo2 fields when `promo2 == 0`); `state_holiday` mixes `0` and `'0'` in the raw file.
