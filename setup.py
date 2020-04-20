"""
See https://docs.pytest.org/en/latest/goodpractices.html#test-discovery
install in editable mode with
pip install -e . --user

Structure should be like:
root
+-- setup.py
+-- mypkg/
|   +-- __init__.py
|   +-- app.py
|   +-- view.py
+-- tests/
|   +-- test_app.py
|   +-- test_view.py
...
"""

from setuptools import setup, find_packages

setup(
    name="JAVER",
    version="0.1",
    author="Atom Scott",
    author_email="atom.james.scott@gmail.com",
    url="https://github.com/AtomScott/japanese-audio-emotion",
    packages=find_packages()
    scripts=[
        "scripts/scrape_and_track"
    ]
    )
