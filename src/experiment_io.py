"""
This file is part of the Relevance Model System.
Copyright (C) 2025, David Schmidig, dave@davencyw.com

The Relevance Model System is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Relevance Model System is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with The Relevance Model System.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import shutil
from datetime import datetime

from tqdm import tqdm

BASE_FOLDER = "results"


def log_to_file(log_file: str, name: str, message: str) -> None:
    """
    Logs a message to the specified file. The message is prepended with the
    current timestamp and the name of the logger.

    Args:
        log_file: The path to the log file.
        name: The name of the logger.
        message: The message to log.
    """
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        f.write(f"[ {name} ] {timestamp} {message}\n")


def make_output_folder(output_folder: str, verbose: bool, remove: bool = False) -> str:
    """
    Creates an output folder for storing results.

    Args:
        output_folder: The path to the output folder.
        verbose: Whether to print progress information.
        remove: Whether to remove the folder if it already exists.

    Returns:
        The path to the output folder.
    """
    output_folder = os.path.join(BASE_FOLDER, output_folder)
    # check if folder exists
    if os.path.exists(output_folder):
        # remove folder if flag set
        if remove:
            shutil.rmtree(output_folder)
            if verbose:
                tqdm.write(f"overwriting output folder: {output_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        if verbose:
            tqdm.write(f"Created output folder: {output_folder}")

        # create log file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file = os.path.join(output_folder, "log.txt")
        with open(file, "w") as fp:
            fp.write(f"[ IO ] {timestamp} Created output folder: {output_folder}\n")

    return output_folder
