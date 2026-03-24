from backend.tools.matching_score import ATSScorer

jd_text = """
We need a graduate engineer trainee for our job.
"""

scorer = ATSScorer()
ranked_candidates = scorer.compute_ats_scores(jd_text)

for c in ranked_candidates:
    print(c["candidate"], c["ats_score"])