import requests
import time


class ResumeMatchAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apyhub.com/utility/sharpapi-resume-job-match-score"

        self.headers = {
            "apy-token": self.api_key
        }

    def submit_job(self, resume_path: str, job_description: str):
        url = f"{self.base_url}/run"

        files = {
            "file": open(resume_path, "rb")
        }

        data = {
            "content": job_description
        }

        response = requests.post(url, headers=self.headers, files=files, data=data)

        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")

        return response.json().get("job_id")

    def get_result(self, job_id: str):
        url = f"{self.base_url}/status/{job_id}"

        while True:
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                raise Exception("Error fetching result")

            data = response.json()

            if data.get("status") == "completed":
                return data.get("result")

            elif data.get("status") == "failed":
                raise Exception("Job failed")

            time.sleep(2)  # wait before retry

    def match(self, resume_path: str, job_description: str):
        job_id = self.submit_job(resume_path, job_description)
        result = self.get_result(job_id)

        return result