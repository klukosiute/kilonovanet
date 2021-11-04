from setuptools import setup, find_packages

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
    long_description=open("README.md").read(),
    install_requires=["numpy", "torch", "pyphot"],
    python_requires='>=3.6'
)
