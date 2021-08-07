from setuptools import setup

setup(
    name="ksm",
    version="0.1.0",
    author="Kamile Lukosiute",
    author_email="lukosiutekamile@gmail.com",
    lisence="MIT",
    packages=["ksm"],
    description="Kilonova surrogate modelling via cVAE",
    long_description=open("README.md").read(),
    install_requires=["numpy", "pytorch"],
)
