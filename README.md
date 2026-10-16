# DIC-2 · Multiple Linear Regression

用多元線性迴歸解兩個問題，流程依照 **CRISP-DM**：

| Problem | Dataset | Target |
|---|---|---|
| 1. Kaggle 50 Startups | [50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv) | `Profit` |
| 2. Boston Housing | [BostonHousing.csv](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) | `medv` |

## 線上 Demo（GitHub Pages）

👉 https://juipiao-chen.github.io/AIoT-DIC-2-Multiple-Linear-Regression/

`index.html` 是一個純前端網頁，所有計算都在瀏覽器裡完成：

- **下拉選單切換兩種 problem**（加分 1）
- **可輸入任意 Dataset URL**（加分 2），也可以上傳本機 CSV；可自選 Target 和 Features，類別欄位會自動做 One-Hot 編碼
- 可選 Backward Elimination（依 p-value）或 All-in，並可調整 significance level、test size、random seed
- 顯示 R²、Adjusted R²、MAE、RMSE（train/test）、迴歸方程式、係數表（SE、t、p-value）、Actual vs Predicted、殘差圖、相關係數圖
- Deployment：輸入特徵值即可得到預測結果

## CRISP-DM

1. **Business Understanding**
   - 50 Startups：創投想知道 R&D、Administration、Marketing 支出與州別中，哪些因素真正影響利潤。
   - Boston Housing：用 13 個社經與環境特徵預測房價中位數（千美元）。
2. **Data Understanding**：敘述統計、缺值檢查、相關係數熱圖（`results/*_corr.png`）。
3. **Data Preparation**：移除缺值、類別欄位 One-Hot（drop first，避免 dummy variable trap）、80/20 train/test split（`random_state=42`）。
4. **Modeling**：OLS 多元線性迴歸，並以 Backward Elimination（SL = 0.05）挑選特徵。
5. **Evaluation**：在 test set 上計算 R²、MAE、RMSE，並畫 Actual vs Predicted（`results/*_pred.png`）。
6. **Deployment**：GitHub Pages 網頁。

## 結果（Python，`random_state=42`）

| Problem | Model | Features | Test R² | Test RMSE |
|---|---|---|---|---|
| 50 Startups | All-in | 5 | 0.8987 | 9055.96 |
| 50 Startups | Backward Elimination | `R&D Spend` | **0.9265** | **7714.33** |
| Boston Housing | All-in | 13 | **0.6688** | **4.93** |
| Boston Housing | Backward Elimination | 10（移除 `zn`、`indus`、`age`） | 0.6523 | 5.05 |

- **50 Startups**：只留下 `R&D Spend` 時，test 表現反而比較好。Administration、Marketing 和 State 在統計上都不顯著，研發支出是影響利潤的主要因素。
- **Boston Housing**：`lstat`（低收入人口比例）、`rm`（房間數）、`ptratio`、`dis`、`nox` 是最重要的特徵。刪掉不顯著的特徵後，模型變簡單，準確度只小幅下降。

> 網頁的切分是在 JS 裡用同一個 seed 做的，跟 sklearn 的切法不同，所以網頁上的數字會和上表有些差異。

## 本機執行

```bash
pip install -r requirements.txt
python mlr_analysis.py                    # 兩個 problem，輸出到 results/
python mlr_analysis.py --problem boston   # 只跑其中一個
python mlr_analysis.py --url <CSV_URL> --target <column>   # 任意資料集

python -m http.server 8000                # 打開 http://localhost:8000 預覽網頁
```

## 部署到 GitHub Pages

1. 把這個資料夾 push 到 GitHub repo 的 `main` branch。
2. 打開 repo 的 **Settings → Pages → Build and deployment → Source**，選 **GitHub Actions**，`.github/workflows/pages.yml` 會自動部署。
   （或者選 *Deploy from a branch*，branch 選 `main`、資料夾選 `/ (root)`。）
3. 網址會是 `https://juipiao-chen.github.io/AIoT-DIC-2-Multiple-Linear-Regression/`。

## 檔案

```
index.html                  GitHub Pages 網頁（MLR web app）
mlr_analysis.py             Python CRISP-DM 分析（兩個 problem / 任意 URL）
data/                       兩個資料集的本機副本
results/                    圖表與 summary.json
.github/workflows/pages.yml GitHub Pages 自動部署
```
