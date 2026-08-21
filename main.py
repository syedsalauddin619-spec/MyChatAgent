import os
import csv
import re
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# RESUME SCREENING AI AGENT
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JD_FILE = os.path.join(BASE_DIR, "job_description.txt")
RESUME_DIR = os.path.join(BASE_DIR, "resumes")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ranked_candidates.csv")

MODEL = "llama3.2:1b"


def read_file(path):
    """Read a text file and return its contents."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def extract_name(text):
    """Extract candidate name from resume text."""
    match = re.search(r"Candidate Name:\s*(.+)", text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return "Unknown Candidate"


def calculate_similarity(jd_text, resume_text):
    """Calculate TF-IDF cosine similarity between JD and resume."""
    documents = [jd_text, resume_text]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


def get_ai_reasoning(jd_text, resume_text, score):
    """Ask Ollama to explain the resume's relevance."""

    prompt = f"""
You are a professional resume screening assistant.

Compare the candidate resume against the job description.

Job Description:
{jd_text}

Candidate Resume:
{resume_text}

TF-IDF similarity score: {score}%

Give a short screening explanation containing:
1. Matching skills
2. Relevant experience or projects
3. Important missing skills
4. Overall suitability

Do not invent information that is not present in the resume.
Keep the explanation concise.
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"].strip()

    except Exception as error:
        return f"AI reasoning unavailable: {error}"


def main():

    print("=" * 60)
    print("        AI RESUME SCREENING AGENT")
    print("=" * 60)

    # Check Job Description
    if not os.path.exists(JD_FILE):
        print("ERROR: job_description.txt was not found.")
        return

    # Check resumes folder
    if not os.path.exists(RESUME_DIR):
        print("ERROR: resumes folder was not found.")
        return

    # Read Job Description
    jd_text = read_file(JD_FILE)

    # Find all TXT resumes
    resume_files = [
        file for file in os.listdir(RESUME_DIR)
        if file.lower().endswith(".txt")
    ]

    if len(resume_files) == 0:
        print("ERROR: No .txt resumes found.")
        return

    print(f"\nFound {len(resume_files)} resumes.")
    print("Screening candidates...\n")

    results = []

    # Process every resume
    for index, filename in enumerate(sorted(resume_files), start=1):

        path = os.path.join(RESUME_DIR, filename)

        resume_text = read_file(path)

        candidate_name = extract_name(resume_text)

        print(
            f"[{index}/{len(resume_files)}] "
            f"Screening {candidate_name}..."
        )

        score = calculate_similarity(
            jd_text,
            resume_text
        )

        reasoning = get_ai_reasoning(
            jd_text,
            resume_text,
            score
        )

        results.append({
            "candidate": candidate_name,
            "resume": filename,
            "score": score,
            "reasoning": reasoning
        })

    # Rank highest score first
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save CSV
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "candidate",
                "resume",
                "similarity_score",
                "reasoning"
            ]
        )

        writer.writeheader()

        for rank, result in enumerate(results, start=1):

            writer.writerow({
                "rank": rank,
                "candidate": result["candidate"],
                "resume": result["resume"],
                "similarity_score": result["score"],
                "reasoning": result["reasoning"]
            })

    # Display ranking
    print("\n" + "=" * 60)
    print("           FINAL CANDIDATE RANKING")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. {result['candidate']} "
            f"- {result['score']}%"
        )

    print("\nScreening completed successfully.")
    print(f"Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
