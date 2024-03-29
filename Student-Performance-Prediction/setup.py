from setuptools import find_packages, setup
from typing import List

HYPON_E_DOT = '- e .'

def get_requirements(requirements_path:str)-> List[str]:
    requirements = []
    with open(requirements_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements if req != HYPON_E_DOT]
    return requirements
setup(
    name='student-performace-prediction',
    version='0.0.1',
    author='muhammad',
    author_email='muhammadof9@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)