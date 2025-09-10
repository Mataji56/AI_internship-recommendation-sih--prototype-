import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


CSV_FILE = "data/internships.csv"

# Load internships
try:
    internships = pd.read_csv(CSV_FILE)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Page config
st.set_page_config(page_title="Find My Internship", layout="wide")

# Background CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1521791136064-7986c2920216");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1 {
    text-align: center;
    font-size: 60px;
    color: #FFD700;
    text-shadow: 2px 2px 5px black;
}
.recommend-card {
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Heading
st.markdown("<h1>Find My Internship</h1>", unsafe_allow_html=True)

# User Inputs
st.subheader("Enter your details 👇")

name = st.text_input("Your Name")
education = st.text_input("Education (e.g. Bachelor, Master, Diploma)")
skills = st.text_input("Skills (comma separated, e.g. Python, SQL, Machine Learning)")
location = st.text_input("Preferred Location (e.g. Bangalore, Delhi, Bhopal)")

if st.button("Get Recommendations"):
    if not skills.strip():
        st.warning("Please enter at least one skill!")
    else:
        # Prepare text data for similarity
        internships["combined"] = (
            internships["skills_required"].fillna('') + " " +
            internships["location"].fillna('') + " " +
            internships["education_required"].fillna('') + " " +
            internships["sector"].fillna('')
        )

        user_input = skills + " " + location + " " + education

        cv = CountVectorizer().fit_transform([user_input] + internships["combined"].tolist())
        similarity_scores = cosine_similarity(cv[0:1], cv[1:]).flatten()

        internships["similarity"] = similarity_scores
        top_recs = internships.sort_values(by="similarity", ascending=False).head(5)

        if name.strip():
            st.subheader(f"🎯 Hello {name}, here are your Recommended Internships:")
        else:
            st.subheader("🎯 Recommended Internships for You:")

        for _, row in top_recs.iterrows():
            st.markdown(f"""
            <div class="recommend-card">
                <h3>{row['title']}</h3>
                <p><b>Company:</b> {row['company']}</p>
                <p><b>Location:</b> {row['location']}</p>
                <p><b>Required Skills:</b> {row['skills_required']}</p>
                <p><b>Sector:</b> {row['sector']}</p>
                <p><b>Education:</b> {row['education_required']}</p>
                <img src="{row['logo_url']}" width="120">
            </div>
            """, unsafe_allow_html=True)

