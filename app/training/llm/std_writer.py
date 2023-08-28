import sys
import re
from app.training.download.progress import update_progress

from app.training.schemas.training import ProgressResponseSchema


class CustomStdErrWriter:
    def __init__(self, repo_id):
        self.original_stderr = sys.stderr
        self.repo_id = repo_id
        sys.stderr = self

    def write(self, msg):
        self.original_stderr.write(f"Print: {msg}")
        progress_response = extract_values(msg, self.repo_id)
        if progress_response.task != "None" and not progress_response.total.startswith(
            "0"
        ):
            update_progress(progress_response)

    def flush(self):
        self.original_stderr.flush()

    def close(self):
        sys.stderr = self.original_stderr


def extract_values(text: str, repo_id: str) -> ProgressResponseSchema:
    percent_pattern = r"(\d+)%"
    total_pattern = r"/(\d+)"
    current_pattern = r"(\d+)/"
    start_time_pattern = r"\[(\d+:\d+)"
    end_time_pattern = r"<(\d+:\d+:\d+)"
    speed_pattern = r"(\d+\.\d+s/it)"

    try:
        curr_percent = int(re.search(percent_pattern, text).group(1))
        total = re.search(total_pattern, text).group(1)
        curr_size = re.search(current_pattern, text).group(1)
        start_time = re.search(start_time_pattern, text).group(1)
        end_time = re.search(end_time_pattern, text).group(1)
        speed = re.search(speed_pattern, text).group(1)

        print(f"curr_percent: {curr_percent}, total: {total}")

        model_instance = ProgressResponseSchema(
            task="training",
            model_name=repo_id,
            total=total,
            curr_size=curr_size,
            curr_percent=curr_percent,
            start_time=start_time,
            end_time=end_time,
            sec_per_dl=speed,
        )

        return model_instance

    except AttributeError:
        print(f"Failed to extract values from text: '{text}'")
        return ProgressResponseSchema.no_progress()

    except Exception as e:
        print(f"Failed to extract values from text due to: {e}")
        return ProgressResponseSchema.no_progress()
