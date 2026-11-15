# Choosing a forecasting method per product for intermittent retail demand (M5)

Code and result files of a master's thesis (Sami Graia, IHEC, Tunisia, 2026) on the
M5 Walmart data: which forecasting method suits which product, and can a simple,
readable rule make that choice?

## What the study does
- Data: the 30,490 product-store series of M5, daily sales, days 1 to 1,913.
- Ten local machine learning models (LightGBM, XGBoost, CatBoost, Gradient Boosting,
  HGBR, Random Forest, Extra Trees, Ridge, Lasso, Elastic Net), tuned per SBC demand
  class with Optuna (TPE).
- Benchmarks: tuned SES, Croston, SBA and TSB, a global LightGBM, the mean and the
  median of the ten models, and a zero forecast.
- Accuracy measure: RMSSE. Test windows W1-W5 (days 1,746-1,885), holdout W6
  (days 1,886-1,913), blind test W7 (days 1,914-1,941).
- Categorization: rules learned on 27,169 learning products and scored on
  3,305 test products.

## Repository contents
- `notebooks/M5_V2_Tuning.ipynb`: tuning of the ten models for each SBC class
  (40 runs)
- `notebooks/M5_V2_Tuning_part2.ipynb`: extension of CatBoost to 75 trials for
  lumpy and erratic products, then the summary of the 40 tuned configurations
  (`optimal_hyperparameters_v2.json` and `Tuning_v2.xlsx`)
- `notebooks/M5_V2_Evaluation.ipynb`: evaluation of all products on windows W1-W6
- `notebooks/M5_V2_Final_Categorization.ipynb`: benchmarks, categorization,
  results, robustness checks and figures
- `src/pipeline_v2.py`: shared definitions (features, windows, RMSSE, classes, models)
- `results/`: the frozen result files used in Chapters 2 and 3 of the thesis
- `inputs/`: the 3,305 sample products (`FINAL_SAMPLE_3305.csv`), the four tuning
  sub-samples (`subsample_*.csv`) with their validation report, and the reference
  classification (`demand_classification_results.csv`). They were built beforehand
  in a separate notebook and are read by the notebooks above.

## Data
The M5 data are not included. Download them from Kaggle:
https://www.kaggle.com/competitions/m5-forecasting-accuracy
Put `sales_train_validation.csv`, `calendar.csv` and `sell_prices.csv` in
`MyDrive/M5_Thesis/data` on Google Drive (`sales_train_evaluation.csv` is used
only for the blind test W7).

## How to run
Before the first run, copy the files of `inputs/` to Google Drive:
`FINAL_SAMPLE_3305.csv` and `demand_classification_results.csv` to
`MyDrive/M5_Thesis/results`, and the four `subsample_*.csv` files to
`MyDrive/M5_Thesis/subsamples`.

Run the notebooks on Google Colab with Google Drive mounted, in this order:
tuning (both parts), evaluation, final. Long cells save their results to Drive and
skip the work when the file already exists. Random seeds are fixed (models and
tuning: 42, sample: 152, other random steps: 0).

## Notes
- Some comments in the notebooks keep earlier wording, for example
  "27,185 learning products". The code uses the 27,169 products that have a test
  result, which are the numbers reported in the thesis.
- The figure cell of the final notebook draws the figures with earlier labels
  and numbering. The figures in the thesis use the final labels.
- Comments in the notebooks use the earlier numbering of the thesis: its
  Chapters 3 and 4 are now Chapters 2 and 3, and the section, table and figure
  numbers have changed.
