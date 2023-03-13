#Used to build an application as package

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str)->List:
    '''This function will return list of requirements'''

    requirements = []
    with open(file_path,'r') as file:
        requirements=file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

setup(
    name="ML-Regression-Project",
    version='0.0.1',
    author='Shashvath',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
