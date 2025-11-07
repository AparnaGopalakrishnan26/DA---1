# Universal Bank - Streamlit Dashboard (Personal Loan Prediction)

This repository contains a Streamlit dashboard to explore the Universal Bank dataset, train three tree-based models (Decision Tree, Random Forest, Gradient Boosting), and predict whether customers will accept a Personal Loan.

## Files (in root, no folders)
- `app.py` - main Streamlit application
- `utils.py` - helper functions for plotting, training, and prediction
- `UniversalBank_sample.csv` - sample dataset for quick demo
- `requirements.txt` - list of packages (no versions) for Streamlit Cloud
- `README.md` - this file

## How to deploy on Streamlit Cloud
1. Create a GitHub repository and push these files at the root (no folders).
2. On Streamlit Cloud, create a new app and connect it to the repository and branch.
3. Use `app.py` as the main file. Streamlit Cloud will install packages from `requirements.txt`.
4. Open the app. Use the sidebar to upload your real `UniversalBank.csv` or use the bundled sample.

## Notes
- The app trains models in memory; for larger datasets, consider adding caching or persisting models.
- The `Predict` tab trains a Random Forest on the sample dataset and applies it to your uploaded file if you haven't trained models previously.
- If you want improved explainability, extend the app to include SHAP visualizations.
