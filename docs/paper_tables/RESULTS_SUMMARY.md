# Block 3 KDD'26 Benchmark Results

**Total records**: 384
**Categories**: ['foundation', 'irregular', 'statistical']
**Tasks**: ['task1_outcome', 'task2_forecast', 'task3_risk_adjust']
**Models**: 9
**Targets**: ['funding_raised_usd', 'investors_count', 'is_funded']


## task1_outcome


### Target: `funding_raised_usd`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Moirai | 691982.79 | 2359979.24 | 107.51 | 8 |
| irregular | SAITS | 1079208.88 | 2413370.49 | 127.57 | 8 |
| irregular | GRU-D | 1079209.27 | 2413370.57 | 127.56 | 8 |
| foundation | Chronos | 7734266.61 | 7955135.87 | 173.86 | 8 |


### Target: `investors_count`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Moirai | 306.48 | 1404.81 | 87.39 | 8 |
| irregular | GRU-D | 311.29 | 1421.35 | 88.84 | 8 |
| irregular | SAITS | 311.39 | 1421.30 | 88.85 | 8 |
| foundation | Chronos | 401.02 | 1395.35 | 105.32 | 8 |


### Target: `is_funded`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Chronos | 0.17 | 0.30 | 28.64 | 8 |
| irregular | SAITS | 0.31 | 0.35 | 47.38 | 8 |
| foundation | Moirai | 0.36 | 0.38 | 55.08 | 8 |
| irregular | GRU-D | 0.42 | 0.44 | 66.39 | 8 |


## task2_forecast


### Target: `funding_raised_usd`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Moirai | 691982.79 | 2359979.24 | 107.51 | 8 |
| irregular | SAITS | 1079208.88 | 2413370.49 | 127.57 | 8 |
| irregular | GRU-D | 1079209.27 | 2413370.57 | 127.56 | 8 |
| foundation | Chronos | 7734266.61 | 7955135.87 | 173.86 | 8 |
| statistical | SF_SeasonalNaive | 12148596.38 | 12266851.02 | 182.39 | 8 |
| statistical | AutoETS | 12248328.14 | 12366054.39 | 182.49 | 8 |
| statistical | AutoARIMA | 12260770.58 | 12378369.77 | 182.51 | 8 |
| statistical | MSTL | 12523452.59 | 12639846.03 | 182.77 | 8 |
| statistical | AutoTheta | 12918871.67 | 13033833.22 | 183.14 | 8 |


### Target: `investors_count`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Moirai | 306.48 | 1404.81 | 87.39 | 8 |
| irregular | GRU-D | 311.29 | 1421.35 | 88.84 | 8 |
| irregular | SAITS | 311.39 | 1421.30 | 88.85 | 8 |
| foundation | Chronos | 401.02 | 1395.35 | 105.32 | 8 |
| statistical | SF_SeasonalNaive | 422.31 | 1397.02 | 108.88 | 8 |
| statistical | MSTL | 425.16 | 1397.24 | 109.23 | 8 |
| statistical | AutoETS | 429.16 | 1397.56 | 109.72 | 8 |
| statistical | AutoTheta | 429.92 | 1397.63 | 109.81 | 8 |
| statistical | AutoARIMA | 431.52 | 1397.77 | 110.00 | 8 |


## task3_risk_adjust


### Target: `funding_raised_usd`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Moirai | 691982.79 | 2359979.24 | 107.51 | 8 |
| irregular | SAITS | 1079208.88 | 2413370.49 | 127.57 | 8 |
| irregular | GRU-D | 1079209.27 | 2413370.57 | 127.56 | 8 |
| foundation | Chronos | 7734266.61 | 7955135.87 | 173.86 | 8 |
| statistical | SF_SeasonalNaive | 12148596.38 | 12266851.02 | 182.39 | 8 |
| statistical | AutoETS | 12248328.14 | 12366054.39 | 182.49 | 8 |
| statistical | AutoARIMA | 12260770.08 | 12378369.28 | 182.51 | 8 |
| statistical | MSTL | 12523452.59 | 12639846.03 | 182.77 | 8 |
| statistical | AutoTheta | 12918871.67 | 13033833.22 | 183.14 | 8 |


### Target: `investors_count`

| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |
|----------|-------|------|-------|--------|---|
| foundation | Moirai | 306.48 | 1404.81 | 87.39 | 8 |
| irregular | GRU-D | 311.29 | 1421.35 | 88.84 | 8 |
| irregular | SAITS | 311.39 | 1421.30 | 88.85 | 8 |
| foundation | Chronos | 401.02 | 1395.35 | 105.32 | 8 |
| statistical | SF_SeasonalNaive | 422.31 | 1397.02 | 108.88 | 8 |
| statistical | MSTL | 425.16 | 1397.24 | 109.23 | 8 |
| statistical | AutoETS | 429.16 | 1397.56 | 109.72 | 8 |
| statistical | AutoTheta | 429.92 | 1397.63 | 109.81 | 8 |
| statistical | AutoARIMA | 431.48 | 1397.77 | 110.00 | 8 |
