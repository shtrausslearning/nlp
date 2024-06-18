
from setuptools import setup, find_packages

setup(packages=find_packages(where="src"),
      package_dir={"": "src"},
      package_data={"mllibs": ["corpus/*.txt",
                               "corpus/*.csv",
                               "*.json",
                               "models/*.pickle"]
      })