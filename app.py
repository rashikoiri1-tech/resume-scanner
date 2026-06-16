from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
#hello
#added new commit

# PATTERN: 15 COMPANIES DATABASE - Easy to add more
COMPANIES_DB = {
    "1. TCS Data Analyst Intern": {
        "desc": "data analysis python sql excel tableau power bi statistics dashboard visualization business intelligence",
        "skills": {"python", "sql", "excel", "tableau", "power bi", "statistics", "data analysis"},
        "type": "Data", "stipend": "10k/month"
    },
    "2. Infosys ML Intern": {
        "desc": "machine learning tensorflow pytorch scikit-learn deep learning ai neural networks python data science",
        "skills": {"python", "tensorflow", "pytorch", "scikit-learn", "machine learning", "deep learning"},
        "type": "AI/ML", "stipend": "15k/month"
    },
    "3. Wipro Web Dev Intern": {
        "desc": "web development html css javascript react angular nodejs express mongodb mysql frontend backend",
        "skills": {"html", "css", "javascript", "react", "nodejs", "mongodb", "web development"},
        "type": "Web Dev", "stipend": "12k/month"
    },
    "4. HCL Java Developer": {
        "desc": "java spring boot hibernate maven j2ee microservices rest api mysql oracle backend development",
        "skills": {"java", "spring boot", "hibernate", "maven", "rest api", "mysql"},
        "type": "Backend", "stipend": "12k/month"
    },
    "5. Tech Mahindra Cloud": {
        "desc": "cloud aws azure gcp docker kubernetes devops linux terraform cloud architecture ci cd",
        "skills": {"aws", "azure", "docker", "kubernetes", "devops", "linux", "cloud"},
        "type": "Cloud", "stipend": "18k/month"
    },
    "6. Accenture Cyber Security": {
        "desc": "cybersecurity network security ethical hacking penetration testing kali linux firewall siem",
        "skills": {"cybersecurity", "network security", "ethical hacking", "linux", "firewall"},
        "type": "Security", "stipend": "15k/month"
    },
    "7. Cognizant Python Dev": {
        "desc": "python django flask fastapi automation scripting oop data structures algorithms",
        "skills": {"python", "django", "flask", "automation", "oops", "data structures"},
        "type": "Python", "stipend": "10k/month"
    },
    "8. IBM Data Science": {
        "desc": "data science python r statistics machine learning jupyter pandas numpy matplotlib watson",
        "skills": {"python", "r", "statistics", "machine learning", "pandas", "numpy"},
        "type": "Data", "stipend": "20k/month"
    },
    "9. Amazon SDE Intern": {
        "desc": "software development java python c++ data structures algorithms system design aws problem solving",
        "skills": {"java", "python", "c++", "data structures", "algorithms", "system design"},
        "type": "SDE", "stipend": "60k/month"
    },
    "10. Microsoft Software": {
        "desc": "software engineering c# dotnet azure typescript react algorithms coding problem solving",
        "skills": {"c#", "dotnet", "azure", "typescript", "react", "algorithms"},
        "type": "SDE", "stipend": "80k/month"
    },
    "11. Google STEP Intern": {
        "desc": "programming python java c++ go data structures algorithms machine learning gcp",
        "skills": {"python", "java", "c++", "data structures", "algorithms", "gcp"},
        "type": "SDE", "stipend": "70k/month"
    },
    "12. Oracle DBA Intern": {
        "desc": "database oracle sql plsql database design performance tuning dba administration",
        "skills": {"oracle", "sql", "plsql", "database", "performance tuning"},
        "type": "Database", "stipend": "15k/month"
    },
    "13. SAP ABAP Developer": {
        "desc": "sap abap hana fiori sapui5 erp mm sd fi co development",
        "skills": {"sap", "abap", "hana", "fiori", "erp"},
        "type": "SAP", "stipend": "18k/month"
    },
    "14. Deloitte Analyst": {
        "desc": "consulting analytics excel powerpoint sql financial modeling business strategy",
        "skills": {"excel", "powerpoint", "sql", "financial modeling", "analytics"},
        "type": "Analyst", "stipend": "25k/month"
    },
    "15. Capgemini Business Analyst": {
        "desc": "business analysis requirements gathering sql jira agile scrum documentation",
        "skills": {"sql", "business analysis", "agile", "documentation", "jira"},
        "type": "Analyst", "stipend": "12k/month"
    }
}

def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

def extract_skills_from_text(text):
    all_skills = set()
    for company in COMPANIES_DB.values():
        all_skills.update(company["skills"])
    found_skills = set()
    for skill in all_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.add(skill)
    return found_skills

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    resume_skills = set()
    
    if request.method == "POST":
        pdf = request.files["resume"]
        if pdf and pdf.filename.endswith(".pdf"):
            resume_text = extract_pdf_text(pdf)
            resume_skills = extract_skills_from_text(resume_text)

            # Core ML Logic: TF-IDF + Cosine Similarity
            all_docs = [resume_text] + [job["desc"] for job in COMPANIES_DB.values()]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_docs)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Calculate results for all 15 companies
            for idx, (company_name, company_data) in enumerate(COMPANIES_DB.items()):
                match_percent = round(similarity_scores[idx] * 100, 1)
                skills_have = resume_skills & company_data["skills"]
                skills_need = company_data["skills"] - resume_skills
                
                results.append({
                    "name": company_name,
                    "type": company_data["type"],
                    "stipend": company_data["stipend"],
                    "score": match_percent,
                    "have": sorted(list(skills_have)),
                    "need": sorted(list(skills_need))
                })
            
            # Sort by score - highest first
            results.sort(key=lambda x: x["score"], reverse=True)

    return render_template("index.html", 
                         results=results, 
                         resume_skills=sorted(list(resume_skills)),
                         total_companies=len(COMPANIES_DB))

if __name__ == "__main__":
    print(f"Evacuation Scanner loaded with {len(COMPANIES_DB)} companies")
    app.run()

