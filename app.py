from flask import Flask, render_template, request
import re
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as stp

app = Flask(__name__)

class ResumeRanker:
    def __init__(self):
        self.stopwords = set(stp.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def lemmatize_text(self, text):
        return " ".join([self.lemmatizer.lemmatize(word) for word in re.findall(r'\w+', text)])
    
    def preprocess_text(self, text):
        text = text.lower()
        text = self.lemmatize_text(text)
        return text
    
    def rank_resumes(self, resumes, query):
        query = self.preprocess_text(query)
        resumes = [self.preprocess_text(resume) for resume in resumes]

        X = self.vectorizer.fit_transform(resumes)
        query_vector = self.vectorizer.transform([query])

        similarity = cosine_similarity(X, query_vector)
        similarity = similarity.flatten()

        ranked_resumes = [(resumes[i], similarity[i], resumes[i]) for i in range(len(resumes))]
        ranked_resumes = sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

        return ranked_resumes

resume_ranker = ResumeRanker()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    resume_paths = request.form['resumefolder'].split(",")
    query = request.form['jobdescription']
    resumes = []
    for path in resume_paths:
        with open(path.strip(), "r", encoding="utf-8") as file:
            resume = file.read()
        resumes.append(resume)
    
    ranked_resumes = resume_ranker.rank_resumes(resumes, query)

    return render_template('result.html', ranked_resumes=ranked_resumes)

if __name__ == '__main__':
    app.run(debug=True)
