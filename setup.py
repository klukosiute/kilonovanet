from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="kilonovanet",
    version="0.1.0",
    author="Kamile Lukosiute",
    author_email="lukosiutekamile@gmail.com",
    lisence="MIT",
    packages=find_packages(exclude=["tests", "notebooks", "data"]),
    classifiers=[
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    description="Kilonova surrogate modelling via cVAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "torch", "pyphot"],
    python_requires='>=3.6'
)
