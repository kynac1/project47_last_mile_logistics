# Installation

The python files are packaged as a module. To install the package and dependencies, running `python -m pip install -e .` from the top level of this repository should suffice.

[Poetry](https://python-poetry.org) was used for dependency management, to simplify locking dependency versions, and to make installation more reproducible. For an install that doesn't touch currently installed python packages, either first create a virtual environment or conda environment, and run the above command, or install poetry and run `poetry install`.

