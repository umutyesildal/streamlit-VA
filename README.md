

# Streamlit Organic Carbon Data Visualization

This project visualizes organic carbon data using a Random Forest classifier in Streamlit.

## Requirements
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage
1. Put your `lucas_organic_carbon_training_and_test_data.csv` and `lucas_organic_carbon_target.csv` in the project folder.  
2. Run:
```bash
streamlit run app.py
```
3. Adjust the test size slider and click **Train Random Forest** to see metrics and plots.

## Features
- Confusion matrix (normalized and standard)  
- Classification report and metrics table  
- PCA scatter plots  
- Feature importances  
- True vs. Predicted distributions

## Notes
- Update paths in 

app.py

 if your data files are located elsewhere.  
- Modify visualization functions or add new ones as needed.  