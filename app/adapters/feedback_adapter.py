import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM = """You are a kind pronunciation teacher for Korean language learners whose native language is Russian.
Keep in mind that Russian speakers often struggle with sounds unique to Korean (ㅓ, ㅡ, final consonants, tense/aspirated consonant distinctions).
Follow these rules:
1. Always start with encouragement.
2. Focus on 1-2 key correction points for the lowest-scoring areas (consonant/vowel/fluency).
3. Explain corrections in simple terms like mouth shape or tongue position.
4. Never use technical terms or IPA symbols.
5. Keep it to 2-3 sentences total."""

def run_feedback(llm_feedback_input: dict) -> str:
    ref_text = llm_feedback_input.get("reference_text", "")
    score = llm_feedback_input.get("score_breakdown", {})
    issues = llm_feedback_input.get("issues", [])

    issue_lines = "\n".join([
        f"- [{issue['severity']}] {issue['description']} (tip: {issue['tip']})"
        for issue in issues[:5]
    ])

    prompt = f"""
Target sentence: {ref_text}
Pronunciation scores (out of 100):
- Overall: {score.get('overall', 0):.1f}
- Consonant accuracy: {score.get('consonant', 0):.1f}
- Vowel accuracy: {score.get('vowel', 0):.1f}
- Final consonant accuracy: {score.get('coda', 0):.1f}
- Fluency: {score.get('fluency_like', 0):.1f} (out of 30)

Main pronunciation errors (most severe first):
{issue_lines}

Please write kind and practical pronunciation feedback in Korean for a Russian-speaking learner.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=SYSTEM + "\n\n" + prompt
    )
    return response.text
