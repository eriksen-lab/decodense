"""
decodense
A Python package for decomposing hf and dft mean-field theory
"""
import sys
from setuptools import setup, find_packages

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name= 'decodense',
    author = 'Dr. Janus Juul Eriksen',
    author_email = 'janus.eriksen@bristol.ac.uk',
    description = short_description[0],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = 'MIT',

    packages=find_packages(),

)
