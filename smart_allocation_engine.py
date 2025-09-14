import pandas as pd
import numpy as np
from collections import Counter
import random
import streamlit as st

# Store company requirements
company_data = []

# Generate Fake Data
def generate_fake_data(num_candidates=500, num_companies=50):
    skills_pool = ["Python", "SQL", "Java", "Excel", "Communication", "Marketing", "Data Analysis", "Web Development"]
    locations = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
    sectors = ["IT", "Finance", "Manufacturing", "Healthcare"]
   
    candidates = [{
        "name": f"Candidate_{i + 1}",
        "skills": random.sample(skills_pool, k=random.randint(2, 4)),
        "location": random.choice(locations),
        "preferred_sector": random.choice(sectors)
    } for i in range(num_candidates)]
   
    companies = [{
        "role": random.choice(["Data Analyst", "Software Developer", "Marketing Intern", "Finance Assistant"]),
        "skills": random.sample(skills_pool, k=random.randint(2, 4)),
        "location": random.choice(locations),
        "sector": random.choice(sectors),
        "slots": random.randint(1, 5)
    } for _ in range(num_companies)]
   
    return pd.DataFrame(candidates), pd.DataFrame(companies)

# Preprocess Skills
def preprocess_skills(df, all_skills):
    vectors = []
    for skills in df['skills']:
        vec = np.zeros(len(all_skills))
        skill_count = Counter(skills)
        for skill, count in skill_count.items():
            if skill in all_skills:
                vec[all_skills.index(skill)] = count
        vectors.append(vec)
    return np.array(vectors)

# Cosine Similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1)[:, np.newaxis]
    norm_b = np.linalg.norm(b, axis=1)[np.newaxis, :]
    return dot_product / (norm_a * norm_b + 1e-8)

# Location Filter
def filter_by_location(candidate_loc, company_loc):
    return candidate_loc == company_loc

# Preference Weighting
def apply_preference_weight(similarity, candidate_pref, company_sector, weight=1.2):
    return similarity * weight if candidate_pref == company_sector else similarity

# AI Matching Engine
def match_candidates_to_companies(companies, input_candidate, top_n=3):
    all_skills = list(set(input_candidate['skills'] + [skill for sublist in companies['skills'] for skill in sublist]))
   
    candidate_vec = preprocess_skills(pd.DataFrame([input_candidate]), all_skills)
    company_vectors = preprocess_skills(companies, all_skills)
   
    sim_scores = cosine_similarity(candidate_vec, company_vectors)[0]
   
    candidate_matches = []
    for j, company in companies.iterrows():
        sim_score = sim_scores[j]
        if not filter_by_location(input_candidate['location'], company['location']):
            continue
        sim_score = apply_preference_weight(sim_score, input_candidate['preferred_sector'], company['sector'])
        candidate_matches.append((j, sim_score, company['role'], company['location'], company['sector']))
    candidate_matches.sort(key=lambda x: x[1], reverse=True)
    return [{"candidate_name": input_candidate['name'], "top_matches": candidate_matches[:top_n]}]

# Streamlit UI with Tabs
st.title("RozgarRahi: AI Internship Allocation Engine")

# Tabs for Candidate and Company
tab1, tab2 = st.tabs(["Candidate Input", "Company Input"])

with tab1:
    st.markdown("Enter your details to find internship matches!")
   
    with st.form("candidate_form"):
        st.header("Candidate Profile")
        name = st.text_input("Name", value="Test Candidate")
        skills = st.text_input("Skills (comma-separated, e.g., Python, SQL)", value="Python, SQL")
        location = st.selectbox("Location", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"])
        sector = st.selectbox("Preferred Sector", ["IT", "Finance", "Manufacturing", "Healthcare"])
        submit = st.form_submit_button("Find Matches")
       
        if submit:
            input_candidate = {
                "name": name,
                "skills": [s.strip() for s in skills.split(",")],
                "location": location,
                "preferred_sector": sector
            }
           
            # Use company_data if available, else generate fake companies
            if company_data:
                companies_df = pd.DataFrame(company_data)
            else:
                _, companies_df = generate_fake_data(num_candidates=0, num_companies=50)
           
            # Run matching
            results = match_candidates_to_companies(companies_df, input_candidate)
           
            st.header("Your Top Internship Matches")
            if results:
                result = results[0]
                st.write(f"**Candidate**: {result['candidate_name']}")
                matches_df = pd.DataFrame([
                    {"Company Role": match[2], "Location": match[3], "Sector": match[4], "Match Score": f"{match[1]:.2f}"}
                    for match in result['top_matches']
                ])
                st.table(matches_df)
                st.success(f"Found {len(result['top_matches'])} {sector} internships in {location}! Apply now!")
            else:
                st.error("No matches found. Try different skills or location.")

with tab2:
    st.markdown("Enter your internship requirements!")
   
    with st.form("company_form"):
        st.header("Company Requirements")
        role = st.text_input("Role", value="Software Developer")
        skills = st.text_input("Required Skills (comma-separated, e.g., Python, Java)", value="Python, Java")
        location = st.selectbox("Location", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"])
        sector = st.selectbox("Sector", ["IT", "Finance", "Manufacturing", "Healthcare"])
        slots = st.number_input("Available Slots", min_value=1, value=3)
        submit = st.form_submit_button("Submit Requirements")
       
        if submit:
            company_entry = {
                "role": role,
                "skills": [s.strip() for s in skills.split(",")],
                "location": location,
                "sector": sector,
                "slots": slots
            }
            company_data.append(company_entry)
            st.success(f"Added {slots} slots for {role} in {location}! Switch to Candidate Input to see matches.")
