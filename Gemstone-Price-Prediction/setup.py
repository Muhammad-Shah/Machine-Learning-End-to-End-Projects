from setuptools import setup, find_packages


def get_requirements(requirements_path: str):
    packages = []
    with open(requirements_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace(
            '\n', '') for requirement in requirements if requirement != '- e .']
        return requirements


setup(
    name='gemstone-price-prediction',
    version='0.0.1',
    author='muhammad',
    author_email='muhammadof9@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
