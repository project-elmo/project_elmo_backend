import sys
import re
import json
from app.training.download.progress import *
from app.training.schemas.training import LoggingResponseSchema, ProgressResponseSchema


class CustomStdErrWriter:
    def __init__(self, repo_id):
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout

        self.repo_id = repo_id
        sys.stderr = self
        sys.stdout = self

        self.prev_log = LoggingResponseSchema(
            task="training",
            model_name=self.repo_id,
            loss="0", 
            learning_rate="0", 
            epoch="0",
        )

    def write(self, msg: str):
        if msg.startswith("{'loss'"):
            valid_json_text = msg.replace("'", '"')
            data_dict=json.loads(valid_json_text)
            
            log = LoggingResponseSchema(
                task="training",
                model_name=self.repo_id,
                loss=str(data_dict['loss']), 
                learning_rate=str(data_dict['learning_rate']), 
                epoch=str(data_dict['epoch']),
                )
            
            if self.prev_log.epoch != log.epoch:
                update_log(log)
                self.prev_log = log

        else:
            progress_response = extract_values(msg, self.repo_id)
            if progress_response.task != "None" and not progress_response.total.startswith(
                "0"
            ):
                update_progress(progress_response)

    def flush(self):
        self.original_stderr.flush()
        self.original_stdout.flush()

    def close(self):
        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout


def extract_values(text: str, repo_id: str) -> ProgressResponseSchema:
    percent_pattern = r"(\d+)%"
    total_pattern = r"/(\d+)"
    current_pattern = r"(\d+)/"
    start_time_pattern = r"\[(\d{1,2}:\d{1,2}:\d{1,2}|\d{1,2}:\d{1,2})"
    end_time_pattern = r"<(\d{1,2}:\d{1,2}:\d{1,2}|\d{1,2}:\d{1,2})"
    speed_pattern = r"(\d+\.\d+s/it)"

    try:
        curr_percent = int(re.search(percent_pattern, text).group(1))
        total = re.search(total_pattern, text).group(1)
        curr_size = re.search(current_pattern, text).group(1)
        start_time = re.search(start_time_pattern, text).group(1)
        end_time = re.search(end_time_pattern, text).group(1)
        speed = re.search(speed_pattern, text).group(1)

        #print(f"curr_percent: {curr_percent}, total: {total}")

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
        #print(f"Failed to extract values from text: '{text}'")
        return ProgressResponseSchema.no_progress()

    except Exception as e:
        print(f"Failed to extract values from text due to: {e}")
        return ProgressResponseSchema.no_progress()
