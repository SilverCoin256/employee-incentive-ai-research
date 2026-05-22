# Dataset

This project uses the IBM HR Analytics Employee Attrition dataset, which is publicly available on Kaggle.

**Download link:**  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

After downloading, place the CSV file here:

```
data/ibm_hr_attrition.csv
```

The dataset has 1,470 rows and 35 columns. It was created by IBM data scientists as a synthetic dataset for benchmarking and demonstration — there is no real employee data in it.

The pipeline auto-detects the column separator. If you get a single-column dataframe on load, it retries with semicolon separation.
