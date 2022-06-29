from setuptools import setup, find_packages

setup(
    name="host_guest_analysis",
    author="Tim Moore and Peter Schwendeman",
    description="Routines for analyzing host–guest systems",
    version="0.0.1",
    packages=find_packages(where=".", exclude=["tests"]),
    install_requires=[
        "setuptools>=45.0",
        "freud-analysis",
        "numpy",
        "rowan",
    ],
)
