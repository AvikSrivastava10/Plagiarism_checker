from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import re

app = Flask(__name__)


def extract_text(file):
    """Extract text from an uploaded file (pdf or txt)."""
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        # read PDF bytes and extract
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return text
    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""


def split_into_sentences(text):
    """
    Very simple sentence splitter.
    Splits on ., ?, ! followed by whitespace (keeps punctuation).
    """
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Split keeping punctuation at end of sentence
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    # remove empty/very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 1]
    return sentences


def sentence_level_matches(sents1, sents2, threshold=0.6):
    """
    Return two boolean lists indicating which sentences are considered matched.
    Uses TF-IDF vectors for sentences and cosine similarity.
    threshold: cosine similarity threshold to mark a match (0..1)
    """
    if not sents1 or not sents2:
        return [False] * len(sents1), [False] * len(sents2)

    # Build TF-IDF on all sentences combined for consistent vector space
    vectorizer = TfidfVectorizer().fit(sents1 + sents2)
    vecs1 = vectorizer.transform(sents1)
    vecs2 = vectorizer.transform(sents2)

    # pairwise cosine similarities
    sim_matrix = cosine_similarity(vecs1, vecs2)  # shape (len(sents1), len(sents2))

    matched1 = [False] * len(sents1)
    matched2 = [False] * len(sents2)

    for i in range(sim_matrix.shape[0]):
        # find best matching sentence in doc2 for sents1[i]
        best_j = sim_matrix[i].argmax()
        best_score = sim_matrix[i][best_j]
        if best_score >= threshold:
            matched1[i] = True
            matched2[best_j] = True

    return matched1, matched2


def highlight_sentences(sentences, matched_flags):
    """
    Return HTML string with matched sentences wrapped in <mark>.
    Non-matched sentences remain plain.
    """
    parts = []
    for sent, matched in zip(sentences, matched_flags):
        safe = sent.replace('\n', ' ')
        if matched:
            parts.append(f"<mark>{safe}</mark>")
        else:
            parts.append(f"{safe}")
    # join with a space/newline for readability in browser
    return " ".join(parts)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check', methods=['POST'])
def check_plagiarism():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if not file1 or not file2:
        return render_template('index.html', error="Please upload both files (.txt or .pdf).")

    text1 = extract_text(file1)
    text2 = extract_text(file2)

    if not text1.strip() or not text2.strip():
        return render_template('index.html', error="One of the files is empty or unsupported type.")

    # Overall document-level similarity
    doc_vectorizer = TfidfVectorizer().fit([text1, text2])
    doc_vecs = doc_vectorizer.transform([text1, text2])
    overall_sim = cosine_similarity(doc_vecs[0:1], doc_vecs[1:2])[0][0]

    # Sentence-level splitting & matching
    sents1 = split_into_sentences(text1)
    sents2 = split_into_sentences(text2)

    # threshold for marking sentences as matched (tunable)
    SENTENCE_MATCH_THRESHOLD = 0.60
    matched1, matched2 = sentence_level_matches(sents1, sents2, threshold=SENTENCE_MATCH_THRESHOLD)

    # Highlight matched sentences
    highlighted1 = highlight_sentences(sents1, matched1)
    highlighted2 = highlight_sentences(sents2, matched2)

    # Simple classification
    if overall_sim >= 0.8:
        verdict = "High similarity (possible plagiarism)"
    elif overall_sim >= 0.5:
        verdict = "Moderate similarity"
    else:
        verdict = "Low similarity"

    # compute a simple sentence-match percentage (how many sentences were matched in doc1)
    sent_match_pct = 0.0
    if sents1:
        sent_match_pct = sum(matched1) / len(sents1)

    return render_template(
        'index.html',
        result=f"{round(overall_sim * 100, 2)}",
        verdict=verdict,
        highlighted1=highlighted1,
        highlighted2=highlighted2,
        sent_match_pct=round(sent_match_pct * 100, 2),
        filename1=file1.filename,
        filename2=file2.filename,
    )


if __name__ == '__main__':
    app.run(debug=True)
