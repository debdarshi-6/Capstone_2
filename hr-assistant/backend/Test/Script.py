import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.resume_parser import parse_resume
from tools.skill_extractor import extract_skills


parsed = parse_resume("C:/Users/User/Downloads/SWETANGSHU_SAHA_FlowCV_Resume_2026-03-24.pdf")

resume_skills = extract_skills(parsed["text"])

print("Extracted Details:", resume_skills)