AI-Powered Data Analyst

Project Overview

AI-Powered Data Analyst is an end-to-end data analytics application
built with Python and Streamlit. It allows users to upload a CSV or
Excel dataset, explore the data through automated profiling and
interactive visualizations, train and compare machine-learning models,
generate data insights, execute natural-language queries, and create a
self-contained HTML report.

The application is designed to bring the major stages of a data-analysis
workflow together in a single interface.

Main Features

1. Dataset Upload and Data Profiling

-   Upload CSV, XLSX, or XLS datasets.
-   Display dataset dimensions and memory usage.
-   Preview the dataset.
-   Display column information and basic statistics.
-   Identify missing values and data characteristics.

2. Exploratory Data Analysis (EDA)

The EDA section provides: - Correlation heatmap. - Univariate
distribution analysis. - Box plots. - Bivariate scatter plots. - Pair
plots and other interactive visualizations where available.

3. Machine Learning

The application supports supervised machine-learning workflows,
including: - Target-column selection. - Classification task
configuration. - Model training. - Model comparison. - Accuracy and
F1-score evaluation. - ROC-AUC evaluation. - Cross-validation results. -
Confusion matrix visualization. - Feature-importance analysis.

For the demonstrated Titanic-like dataset, the application compared: -
Logistic Regression - Random Forest - XGBoost

The demonstrated run selected Random Forest as the best model according
to the application's comparison results.

4. AI Insights and Natural-Language Queries

-   Ask questions about the uploaded dataset using a natural-language
    query interface.
-   Display deterministic results computed from the data.
-   Provide an AI-generated narrative when the required IBM watsonx.ai
    credentials are configured.

5. Report Generation

-   Generate a self-contained HTML report.
-   Include dataset profile, EDA findings, machine-learning results,
    insights, and natural-language query results.
-   Preview and download the generated report.

Technology Stack

-   Python
-   Streamlit
-   Pandas
-   NumPy
-   Scikit-learn
-   Plotly
-   XGBoost
-   IBM watsonx.ai integration
-   HTML reporting

Project Structure

``` text
AI_Data_Analyst/
├── assets/
├── datasets/
├── screenshots/
├── src/
├── tests/
├── .venv/
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .gitignore
├── PROJECT_PLAN.md
├── AGENTS.md
└── Rachana_AI_Data_Analyst.py
```

Running the Application

1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_Data_Analyst
```

2. Create and activate a virtual environment

Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies

``` bash
pip install -r requirements.txt
```

4. Run the Streamlit application

If the main application file is in the project root:

``` bash
streamlit run Rachana_AI_Data_Analyst.py
```

If the application entry point is inside `src/`, use the corresponding
Streamlit entry file from that directory.

5. Open the application

After Streamlit starts, open the local URL shown in the terminal,
normally:

``` text
http://localhost:8501
```

Usage Workflow

1.  Open the Streamlit application.
2.  Upload a CSV or Excel dataset.
3.  Review the dataset preview and column information.
4.  Open the EDA section and examine the visualizations.
5.  Select the target column and task type for machine learning.
6.  Train the available models.
7.  Compare the model metrics.
8.  Review the confusion matrix and feature importance.
9.  Use the Insights section to ask questions about the dataset.
10. Generate and download the HTML report.

Demonstrated Dataset

The application was demonstrated using:

``` text
titanic_like.csv
```

The demonstrated dataset contains 300 rows and 7 columns.

Screenshots

The repository includes screenshots documenting the main application
outputs, including: - Home/welcome screen - Dataset upload - Dataset
preview - Column information - EDA correlation heatmap - EDA univariate
visualization - EDA bivariate visualization - Model comparison - Model
evaluation metrics - Confusion matrix - Feature importance - AI
Insights - Natural-language query result - Report generation - Generated
report preview

Testing

The repository contains a `tests/` directory for project tests. Install
the development dependencies when required:

``` bash
pip install -r requirements-dev.txt
```

Run tests with:

``` bash
pytest
```

IBM watsonx.ai Configuration

The application can use IBM watsonx.ai for AI-generated narrative
insights. When enabled, the required environment variables should be
configured securely rather than hard-coded in source files.

The application can still display deterministic data-driven results when
the watsonx.ai credentials are not configured.

Output

The application produces: - Interactive EDA visualizations. -
Machine-learning evaluation results. - Confusion matrix and
feature-importance visualizations. - Data-query results. - A
self-contained HTML analysis report.

Author

**Koppavarapu Lakshmi Rachana**

M.Tech student in Data Science with a background in Computer Science and
Engineering.

Notes

-   Do not commit passwords, API keys, access tokens, or other secrets
    to the repository.
-   Keep the virtual environment (`.venv`) out of version control.
-   Keep generated files and temporary outputs separate from source code
    where appropriate.
