def validate_job_description(text: str) -> bool:
    """True if it has enough content to be evaluated as a JD."""
    return len(text.split()) > 30

def validate_resume(text: str) -> bool:
    """True if it contains typical resume identifiers."""
    return len(text.split()) > 20
