from setuptools import setup, find_packages

setup(
    name="gpi-pusht",
    version="0.1.0",
    description="GPI policies and PushT dynamics/vision code",
    packages=find_packages(include=["gpi", "pusht", "pusht_dynamics"]),
)

