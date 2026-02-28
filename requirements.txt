============================================================================================================================================
Multi-label Toxic Comment Detection using Supervised & Unsupervised Learning

Course: Text Mining and Search (A.Y. 2025/2026)
Authors: Any Das (922710) & Aarohi Rakeshkumar Mistry (925352)
Institution: Università degli Studi di Milano-Bicocca
Development Platform: Kaggle Kernels
============================================================================================================================================

1. Project Overview:
This project tackles multi-label toxic comment detection using the Jigsaw dataset. 
It combines supervised classification (TF-IDF with LinearSVC & Ridge, Text CNN with FastText, DistilBERT) and 
unsupervised clustering (TF-IDF + Truncated SVD + K-Means/HDBSCAN, SBERT + UMAP + K-means/HDBSCAN) to identify toxic patterns and subcultures.

-----------------------------------------------------------------------------------------------------------------------------------------------

2. Project Structure (Google Drive):
The submitted folder is organized as follows:

TMS_Project_Das_Mistry/
|
├── Notebooks/                                   		# Contains the 3 Jupyter Notebooks (.ipynb)
│   ├── Text_Classification_Clustering_TF-IDF.ipynb         	# Statistical Baseline TF-IDF (LinearSVC, Ridge, K-Means, HDBSCAN))
│   ├── Text_Classification_FastText_DistillBERT.ipynb      	# Deep Learning (Text CNN) & DistilBERT
│   └── Text_Clustering_SBERT.ipynb                         	# Semantic Clustering (SBERT + HDBSCAN)
|
├── Datasets/                                              	# Contains Jigsaw Toxic Comment dataset & FastText embeddings
│   ├── crawl-300d-2M.vec.zip
│   ├── jigsaw-toxic-comment-classification-challenge.zip
|
├── Report/
│   └── TM&S_Report.pdf                                    	# Detailed Scientific Report of the project
|
├── Presentation/
│   └── TM&S_Presentation.pdf                              	# Project Presentation Slides
|
└── README.txt                                             	# Project Documentation (This file)

-----------------------------------------------------------------------------------------------------------

3. How to Run the Code (Kaggle Instructions):

Step 1: Setup
1. Download the .ipynb files from the Code/ directory of this submission.
2. Log in to Kaggle (https://www.kaggle.com/).
3. Create a new Notebook or use the "Import Notebook" feature to upload the .ipynb files one by one.

Step 2: Add Datasets
The code relies on the Jigsaw Toxic Comment Classification Challenge dataset and FastText Embeddings. You can add them using Option A (Recommended) or Option B (Manual Upload).

Option A: Add via Kaggle Search (Fastest & Recommended)
1. Open the notebook in Kaggle.
2. Click "Add Data" (or "+" icon) in the right sidebar.
3. For Jigsaw Dataset: Search for "Jigsaw Toxic Comment Classification Challenge" and click "+" to add.
4. For FastText: Search for "fasttext-crawl-300d-2m" and click "+" to add (Required for Deep Learning notebook).

Option B: Upload from Drive (Manual)
1. Download the dataset files from the Drive folder to your local machine.
2. In the Kaggle Notebook, click "Add Data" -> "Upload" (top right corner of the popup).
3. Drag and drop the downloaded files to upload them.
4. Important Note: If you upload manually, Kaggle may assign a custom path to the dataset. You might need to update the file paths in the code (e.g., pd.read_csv('../input/your-dataset-name/train.csv')) to match your uploaded folder name.

Step 3: Install Libraries & Setup
The notebooks already contain the necessary commands to set up the environment. The specific requirements are listed below for reference.

   [Code Snippet embedded in Notebooks]
   # Install essential packages and fixes
   !pip install protobuf==3.20.3 --no-dependencies --force-reinstall
   !pip install sentence-transformers
   !pip install umap-learn==0.5.3
   !pip install hdbscan
   !pip install pandas numpy scipy scikit-learn matplotlib seaborn nltk wordcloud torch transformers fasttext tqdm

   # NLTK additional downloads
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')

   # Environment setup
   import os
   os.environ["TOKENIZERS_PARALLELISM"] = "false"

   -----------------------------------------------------------

Required Libraries (installable via pip):
   - pandas, numpy, scipy, scikit-learn
   - matplotlib, seaborn, wordcloud
   - nltk, tqdm
   - torch, transformers, sentence-transformers, fasttext
   - hdbscan, umap-learn==0.5.3
   - protobuf==3.20.3

   Built-in Python modules (no need to install):
   - os, re, zipfile, random, time, collections, itertools, warnings

Step 4: Execution
1. Enable GPU: For the 'Text_Classification' and 'Text_Clustering' notebooks, go to Settings -> Accelerator -> Select "GPU T4 x2".
2. Run All: Click the "Run All" button. All necessary libraries will be installed automatically via the commands above.
======================================================================================================================================

4. REFERENCES
- Dataset: Jigsaw Toxic Comment Classification Challenge (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Pre-trained Embeddings:
  - FastText (crawl-300d-2M.vec)
  - Sentence-BERT (all-MiniLM-L6-v2)
- Models & Algorithms:
  - Classification: DistilBERT, Text CNN, LinearSVC, Ridge Classifier
  - Clustering & Dimensionality Reduction: HDBSCAN, K-Means, UMAP, Truncated SVD (LSA)

=======================================================================================================================================