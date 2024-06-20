#  _______________________________
# / Don't want to self-host?      \
# \ Try .json at http://dottxt.co /
#  -------------------------------
#        \   ^__^
#         \  (oo)\_______
#            (__)\       )\/\
#                ||----w |
#                ||     ||
#
#
# Copyright 2024- the Outlines developers
# Copyright 2023 the vLLM developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from fastapi_serve import TIMEOUT_KEEP_ALIVE

import os
from multiprocessing import cpu_count

def number_of_workers() -> int:
    """Calculate the number of workers based on CPU count."""
    return cpu_count() * 2 + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--workers", type=int, default=number_of_workers())
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Adds the `engine_use_ray`,  `disable_log_requests` and `max_log_len`
    # arguments
    engine_args: AsyncEngineArgs = AsyncEngineArgs.from_cli_args(args)  # type: ignore

    # Sets default for the model (`facebook/opt-125m`)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    command = (
        f"gunicorn -w {args.workers} -k uvicorn.workers.UvicornWorker "
        f"--bind {args.host}:{args.port} "
        f"--timeout-keep-alive {TIMEOUT_KEEP_ALIVE} "
    )

    if args.ssl_keyfile and args.ssl_certfile:
        command += (
            f"--keyfile {args.ssl_keyfile} "
            f"--certfile {args.ssl_certfile} "
        )

    command += "fastapi_serve:app"
    os.system(command)
