from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

app = Flask(__name__)
model = joblib.load('job1.pkl')

# Load job data from the CSV file
job_data = pd.read_csv('IT_salaries.csv')  # Update the file name if needed

# Fill NaN values in the 'key_skills' column with an empty string
job_data['key_skills'] = job_data['key_skills'].fillna('').str.lower()

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(job_data['key_skills'])

def process_job_title(job_title):
    # Check if job_title is a valid string, if not return an empty string
    if isinstance(job_title, str):
        # Clean job title (remove extra spaces and make it lowercase)
        cleaned_title = " ".join(job_title.split()).lower().replace(" ", "")
        # Check if the cleaned title is not empty and has a minimum length
        if cleaned_title and len(cleaned_title) > 2:
            # Also consider the reversed order of 'machinelearning' and 'python'
            reversed_title = cleaned_title.replace("machinelearning", "python").replace("python", "machinelearning")
            return (cleaned_title, reversed_title)
    return ("", "")

def format_job_title(job_title):
    # Add spaces between words in a job title
    return ' '.join(job_title.split('-'))

@app.route('/')
def index():
    return render_template('index.html', recommended_jobs=[])

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        input_skills = process_job_title(request.form.get('skills'))
        input_skills_tfidf = tfidf_vectorizer.transform([input_skills[0], input_skills[1]])
        cosine_scores = cosine_similarity(input_skills_tfidf, tfidf_matrix)
        job_indices = cosine_scores.argsort()[0][::-1]

        recommended_jobs_set = set()
        fuzzy_threshold = 80

        for index in job_indices:
            job_title = format_job_title(process_job_title(job_data['job_title'].iloc[index])[0])
            reversed_job_title = format_job_title(process_job_title(job_data['job_title'].iloc[index])[1])

            if job_title and fuzz.partial_ratio(job_title, input_skills[0]) >= fuzzy_threshold:
                recommended_jobs_set.add(job_title)
            elif reversed_job_title and fuzz.partial_ratio(reversed_job_title, input_skills[1]) >= fuzzy_threshold:
                recommended_jobs_set.add(reversed_job_title)

            # Break the loop once the top 5 recommendations are found
            if len(recommended_jobs_set) == 5:
                break

        recommended_jobs = sorted(list(recommended_jobs_set))

        return render_template('index.html', recommended_jobs=recommended_jobs)

if __name__ == '__main__':
    app.run(debug=True)
