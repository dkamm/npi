from setuptools import setup

setup(
    name='npi',
    version='0.0.1',
    description='Neural Programmer Interpreter implementation',
    author='David Kamm',
    author_email='davidfkamm@gmail.com',
    install_requires=[
        'numpy',
        'pytest',
        'tensorflow',
        'tqdm',
    ],
)
