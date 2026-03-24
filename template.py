import os 
from pathlib import Path

project_name = "hr-assistant"

list_of_files = [

    f"{project_name}/__init__.py",

    f"{project_name}/backend/__init__.py",
    f"{project_name}/backend/main.py",

    f"{project_name}/backend/tools/__init__.py",
    f"{project_name}/backend/tools/resume_parser.py",
    f"{project_name}/backend/tools/skill_extractor.py",
    f"{project_name}/backend/tools/matching_score.py",
    f"{project_name}/backend/tools/ranking.py",

    f"{project_name}/backend/rag/__init__.py",
    f"{project_name}/backend/rag/vector_store.py",
    f"{project_name}/backend/rag/retriever.py",

    f"{project_name}/backend/memory/__init__.py",
    f"{project_name}/backend/models/__init__.py",

    f"{project_name}/frontend/__init__.py",
    f"{project_name}/frontend/app.py",

    f"{project_name}/evaluation/rag_tests.ipynb",
    f"{project_name}/evaluation/bleu_rouge.ipynb",
    f"{project_name}/evaluation/latency_tests.ipynb",

    f"{project_name}/data/job_descriptions/.gitkeep",
    f"{project_name}/data/resumes/.gitkeep",
    f"{project_name}/data/skill_frameworks/.gitkeep",
]

# Create files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"File already exists at: {filepath}") 

        