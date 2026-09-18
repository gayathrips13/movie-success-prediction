
MOVIE SUCCESS PREDICTION — PYTHON + MACHINE LEARNING WEBSITE

FEATURES
- Flask Python backend
- Pandas data processing
- Scikit-learn GradientBoostingClassifier
- Automatic CSV upload and model training
- Movie success prediction page
- Analytics dashboard
- Accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix data
- Feature importance
- Genre and release-year charts
- Top-rated movie table
- Dataset preview
- JSON prediction API
- Responsive dark/pink/purple UI

DATASET
Upload your CSV from the Home page. The app expects these logical fields:
- Release Year / Released_Year
- Runtime
- No_of_Votes / Votes
- Gross
- IMDB_Rating / Rating
Optional:
- Series_Title / Title
- Genre
- Director
- Overview
- Certificate

TARGET
IMDb Rating >= 8.0 -> Successful (1)
IMDb Rating < 8.0 -> Not Successful (0)

FEATURES USED
- Release Year
- Runtime
- log(Number of Votes)
- log(Gross Revenue)

MODEL
GradientBoostingClassifier with:
n_estimators=180
learning_rate=0.05
max_depth=3
min_samples_split=5

RUN
1. Install Python 3.10+ (3.11/3.12 recommended if your environment has package compatibility issues).
2. Open a terminal in this project folder.
3. Create a virtual environment:
   python -m venv venv
4. Activate:
   Windows: venv\Scripts\activate
   macOS/Linux: source venv/bin/activate
5. Install:
   pip install -r requirements.txt
6. Run:
   python app.py
7. Open:
   http://127.0.0.1:5000

NETLIFY
This version uses a Python Flask backend, so it cannot be deployed as a plain static Netlify drag-and-drop site. Use a Python-capable host for the full ML app (for example Render, Railway, PythonAnywhere, or a VPS). Netlify can host only the frontend unless you connect it to a separate backend/API.

ACADEMIC NOTE
Because votes and gross are post-release performance signals, this is best presented as an ML-based movie success assessment using performance features. For a true pre-release success forecast, use only features available before release and define a target such as later box-office success.
