# 🧠 Plag – Python-Based Plagiarism Checker

A simple yet effective **text similarity and plagiarism detection tool** built with Python.  
It compares multiple `.txt` files in a folder and calculates pairwise similarity using vector-based methods.

---

## 🚀 Features

- 🔍 Detects similarity between `.txt` files  
- 📊 Outputs a detailed similarity matrix in CSV format  
- ⚙️ Uses **TF-IDF Vectorization** and **Cosine Similarity**  
- 🧩 Can be extended to use DSA-based algorithms like:
  - Rabin–Karp (String Matching)
  - Longest Common Subsequence (LCS)
  - Hashing and Frequency Maps

---

## 🗂️ Folder Structure

plag/
│
├── app.py
├── README.md
├── templates/
│ └── index.html
└── uploads/


---

## 💻 Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/plag.git
   cd plag


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows


Install dependencies

pip install flask scikit-learn


Run the app

python app.py


Open in browser

http://127.0.0.1:5000/
