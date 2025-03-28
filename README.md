[![license](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit/)
[![Tests](https://github.com/omar-abdelgawad/python-project-template/actions/workflows/tests.yml/badge.svg)](https://github.com/omar-abdelgawad/python-project-template/actions)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

# Project Name
This is a modern template for a python project with the pyproject.toml with some fields to change based on project. It also has tox configured, docs dir for github pages, .github dir with tox-gh-actions configured and more.  
- pyproject.toml
- setuptools backend (setup.py and setup.cfg)
- the src layout
- tox configured
- docs dir for github pages with a tests.yml ready
- .github dir with a tests.yml configured with tox-gh-actions
- tox configured with pytest, ruff, and mypy
- tests dir for pytest 

Please Note that tox is configured with python 3.8,3.9, and 3.10. Feel free to reconfigure everything to your need.

## Installation/Usage
```bash
 $ git clone <repo_url>
 $ cd <repo_name>
 $ virtualenv venv
 $ source venv/bin/activate
 $ pip install -e ".[dev]"
``` 

## Testing with tox
Just running `tox` with no args should work.
```bash
 $ tox
```
tox creates virtual environments and runs all of pytest, ruff, and mypy.
## Package(s)

### package1
....
### package2
....
