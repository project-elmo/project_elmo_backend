import os
import re

from subprocess import Popen, PIPE
from transformers import AutoTokenizer

from app.training.download.progress import (
    update_progress,
)
from app.training.schemas.training import ProgressResponseSchema
from core.config import config


def hub_download(repo_id: str):
    script_directory = os.path.dirname(os.path.abspath(__file__))

    cmd = ["python", f"{script_directory}/run_load_model.py", repo_id]

    proc = Popen(
        cmd,
        stderr=PIPE,
        universal_newlines=True,
    )
    while proc.poll() is None:
        line = proc.stderr.readline()
        print("Print:" + line)
        progress_response = extract_values(line, repo_id)

        if progress_response.task != "None" and not progress_response.total.startswith(
            "0"
        ):
            update_progress(progress_response)


def extract_values(text: str, repo_id: str) -> ProgressResponseSchema:
    """
    Extract the progress from tqdm message while downloading the pre-trained model
    """
    # Downloading model.safetensors:  63%|██████▎   | 346M/548M [00:32<00:19, 10.5MB/s]
    percent_pattern = r"(\d+)%"
    curr_size_pattern = r"(\d+[a-zA-Z]+)"
    total_size_pattern = r"/(\d+\s*[a-zA-Z]+)"
    start_time_pattern = r"\d+:\d+"  # 04:58
    end_time_pattern = r"<(\d+:\d+)"  # <01:45
    speed_pattern = r"(\d+\w+/s)"  # 917kB/s

    try:
        # Extract values
        curr_percent_matches = re.findall(percent_pattern, text)
        curr_size_matches = re.search(curr_size_pattern, text)
        total_size_matches = re.search(total_size_pattern, text)
        start_time_match = re.search(start_time_pattern, text)
        end_time_match = re.search(end_time_pattern, text)
        speed_match = re.search(speed_pattern, text)

        curr_percent = int(curr_percent_matches[0])
        curr_size = curr_size_matches.group(0)
        total = total_size_matches.group(0)[1:]
        start_time = start_time_match.group(0)
        end_time = end_time_match.group(1)
        sec_per_dl = speed_match.group(1)

        print(f" curr_percent: {curr_percent}, curr_size: {curr_size}, total: {total}")

        model_instance = ProgressResponseSchema(
            task="downloading",  # [downloading, training, None]
            model_name=repo_id,
            total=total,
            curr_size=curr_size,
            curr_percent=curr_percent,
            start_time=start_time,
            end_time=end_time,
            sec_per_dl=sec_per_dl,
        )

        return model_instance

    except AttributeError:
        print(f"Failed to extract values from text: '{text}'")
        return ProgressResponseSchema.no_progress()

    except Exception as e:
        print(f"Failed to extract values from text due to: {e}")
        return ProgressResponseSchema.no_progress()


# TODO
def get_tokenizer(repo_id, path):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.save_pretrained(save_directory=path)
