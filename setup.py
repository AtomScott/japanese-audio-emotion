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

setup(name="JVAER", packages=find_packages())
