# PropertyComps

### Folder structure (not mentioned files are out of scope)
best-property/
├── .venv/                     # Python virtual environment
├── api/                       # API implementation
│   ├── main.py               # FastAPI application entry point
│   ├── endpoints/            # API route definitions
│   │   ├── appraisal.py     # Property appraisal endpoint
│   │   └── property.py      # Property management endpoints
│   │
│   ├── services/            # Business logic services
│   │   ├── appraisal_service.py      # appraisal logic (including comps selection)
│   │   ├── training_service.py       # ML model training and updates
│   │   ├── property_service.py       # Property CRUD operations
│   │
│   └── utils/               # API-specific utilities
│
├── data/                     # Data storage directory
│
├── data-sample/              # Data samples to used when testing the APIs
│
├── models/                   # Machine learning models
│   ├── knn_model.joblib     # K-Nearest Neighbors model for property matching
│   ├── cluster_model.joblib # Clustering model for property grouping
│   └── vectorizer.joblib    # Text vectorizer for property descriptions
│
├── notebooks/               # Jupyter notebooks for analysis and development (for testing)
│
├── utils/                   # General utility functions
│
├── requirements.txt            # Python package dependencies
└── README.md                  # Project documentation

### Use case flow
Step 1. Call Add Properties post api (/api/properties), notice the followings:
        - Add sample properties(json array) to the body
        - Set optionaly the 'retrain' to true, to retrain the ML models

Step 2. Call Post Appraisal api (/api/appraisals), check the followings:
        - Add sample subject (appraisal property) and properties (candidate comps)
        - The response is newly created appraisal id and status
        - Comps are being automatically set after creating the appraisal

Step 3. Call Get Appraisal api (/api/appraisals/{appraisal_id}), check the followings:
        - The reponse contains the appraisal data including the selected 3 comps

# NOTE: you can find sample data in the folder data_sample/ 

