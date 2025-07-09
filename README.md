# 🧠 Plagiarism Detector

The **Plagiarism Detector** project is a comprehensive solution for detecting plagiarism and identifying similarities between text documents. Leveraging Natural Language Processing (NLP), web scraping, and advanced data visualization, this tool enables users to easily analyze documents for originality through a streamlined Streamlit-based web interface.

---

## ✨ Features

- 🌓 Light/Dark mode toggle
- 🖥️ Fullscreen viewing experience
- 📄 Flexible input (text or file upload)
- 📊 Advanced similarity measurement using NLP
- 🌐 Web scraping to compare with online sources
- 📈 Interactive visualizations (Plotly)
- 🧠 Tokenization and cosine similarity calculations
- ⚙️ Simple and intuitive UI (Streamlit)

---

## 🛠️ Tech Stack

**Client:**  
- Streamlit  
- HTML/CSS

**Server:**  
- Python

**Libraries Used:**  
- `pandas`  
- `nltk` (Natural Language Toolkit)  
- `BeautifulSoup4`  
- `scikit-learn` (`CountVectorizer`, `cosine_similarity`)  
- `docx2txt`  
- `PyPDF2`  
- `plotly.express`

**Algorithms:**  
- Tokenization  
- Cosine Similarity  
- Web Scraping  
- Document Retrieval  
- Data Visualization

---

## 🚀 Usage / Examples

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(text1, text2):
    text_list = [text1, text2]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    similarity = cosine_similarity(count_matrix)[0][1]
    return similarity

def get_similarity_list(texts, filenames=None):
    similarity_list = []
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list
```

# Clone the repository
git clone https://github.com/CodeCrusaderX/PlagiarismDetection.git

# Navigate into the project directory
cd PlagiarismDetection

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py


👨‍💻 Author
@CodeCrusaderX
