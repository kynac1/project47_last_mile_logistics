# Installation

The python files are packaged as a module. To install the package and dependencies, running `python -m pip install -e .` from the top level of this repository should suffice.

[Poetry](https://python-poetry.org) was used for dependency management, to simplify locking dependency versions, and to make installation more reproducible. For an install that doesn't touch currently installed python packages, either first create a virtual environment or conda environment, and run the above command, or install poetry and run `poetry install`.

## Tests

To check that everything is installed properly, we have a unit-testing suite. This requires the OSRM server to be started. Run `pytest tests` to run all the tests. Pytest should be installed as a development dependency; if not, it can easily be installed with `pip install pytest` or similar.