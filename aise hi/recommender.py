import pandas as pd

def load_data(path="data/internships.csv"):
    return pd.read_csv(path)

def recommend_internships(user_skills, user_education, user_location, internships):
    skills_list = [s.strip().lower() for s in user_skills.split(",")]

    def match_score(row):
        internship_skills = str(row['skills_required']).lower().split(";")
        score = len(set(skills_list) & set(internship_skills))
        # +1 if location matches
        if user_location.lower() in str(row['location']).lower():
            score += 1
        # +1 if education matches
        if user_education.lower() in str(row['education_required']).lower():
            score += 1
        return score

    internships['score'] = internships.apply(match_score, axis=1)
    return internships.sort_values(by="score", ascending=False).head(5)
