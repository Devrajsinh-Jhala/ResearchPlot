from setuptools import setup, find_packages

setup(
    name='researchplot',
    version='0.1.0',
    description='A package to create publication-ready plots for researchers.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)