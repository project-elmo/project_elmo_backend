import os
import copy
import logging
from typing import BinaryIO, Dict, Optional
from urllib.parse import urlparse

import requests
from requests import Response
from requests.exceptions import ProxyError, Timeout
from transformers import AutoTokenizer
from tqdm import tqdm

from huggingface_hub.file_download import (
    logger,
    http_backoff,
    hf_hub_url,
    HTTP_METHOD_T,
    HEADER_FILENAME_PATTERN,
)
from app.training.download.progress import (
    initialize_progress,
    update_progress,
    reset_progress,
)


def download(model_name: str, namespace: str = None, path: str = None):
    """Download pre-trained model from huggingface."""
    user_home = os.getcwd()
    default_path = os.path.join(user_home, "elmo", "models")
    path = os.path.join(default_path, model_name) if not path else path

    filename = "pytorch_model.bin"
    file_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)

    repo_id = f"{namespace}/{model_name}" if namespace else model_name

    url_to_download = hf_hub_url(repo_id, filename, repo_type="model")
    response = get_response(url_to_download)
    expected_size = int(get_file_size(response))

    if os.path.exists(file_path) and os.path.getsize(file_path) == expected_size:
        print(f"{file_path} already downloaded")
        return

    with open(file_path, "wb") as file:
        http_get_file(url_to_download, file, repo_id, response, expected_size)


def get_tokenizer(repo_id, path):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.save_pretrained(save_directory=path)


def get_response(url):
    response = request_wrapper(
        method="GET", url=url, stream=True, headers={}, timeout=10000.0, max_retries=1
    )

    return response


def get_file_size(response: Response) -> str:
    return response.headers.get("Content-Length")


def get_displayed_name(content_disposition: Optional[str], url: str) -> str:
    if content_disposition:
        match = HEADER_FILENAME_PATTERN.search(content_disposition)
        if match:
            displayed_name = match.groupdict()["filename"]
            return (
                f"(…){displayed_name[-20:]}"
                if len(displayed_name) > 22
                else displayed_name
            )
    return url


def http_get_file(
    url: str,
    temp_file: BinaryIO,
    model_name: str,
    response: Response,
    expected_size: int,
):
    headers = {"Range": f"bytes={temp_file.tell()}-"}

    displayed_name = get_displayed_name(
        response.headers.get("Content-Disposition"), url
    )

    initialize_progress(model_name, expected_size)

    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=expected_size,
        desc=f"Downloading {displayed_name}",
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    )

    for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
            update_progress(model_name, len(chunk))

    progress.close()
    reset_progress(model_name)


def request_wrapper(
    method: HTTP_METHOD_T,
    url: str,
    *,
    max_retries: int = 0,
    base_wait_time: float = 0.5,
    max_wait_time: float = 2,
    timeout: Optional[float] = 10.0,
    follow_relative_redirects: bool = False,
    **params,
) -> requests.Response:
    if follow_relative_redirects:
        response = request_wrapper(
            method=method,
            url=url,
            max_retries=max_retries,
            base_wait_time=base_wait_time,
            max_wait_time=max_wait_time,
            timeout=timeout,
            follow_relative_redirects=False,
            **params,
        )

        # If redirection, we redirect only relative paths.
        # This is useful in case of a renamed repository.
        if 300 <= response.status_code <= 399:
            parsed_target = urlparse(response.headers["Location"])
            if parsed_target.netloc == "":
                # This means it is a relative 'location' headers, as allowed by RFC 7231.
                # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
                # We want to follow this relative redirect !
                #
                # Highly inspired by `resolve_redirects` from requests library.
                # See https://github.com/psf/requests/blob/main/requests/sessions.py#L159
                return request_wrapper(
                    method=method,
                    url=urlparse(url)._replace(path=parsed_target.path).geturl(),
                    max_retries=max_retries,
                    base_wait_time=base_wait_time,
                    max_wait_time=max_wait_time,
                    timeout=timeout,
                    follow_relative_redirects=True,  # resolve recursively
                    **params,
                )
        return response

    return http_backoff(
        method=method,
        url=url,
        max_retries=max_retries,
        base_wait_time=base_wait_time,
        max_wait_time=max_wait_time,
        retry_on_exceptions=(Timeout, ProxyError),
        retry_on_status_codes=(),
        timeout=timeout,
        **params,
    )
