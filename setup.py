from setuptools import setup, find_packages

setup(
    name='researchplot',
    version='0.1.1',
    description='A package to create publication-ready plots for researchers.',
    author='Devrajsinh Jhala',
    author_email='jhaladevrajsinh11@gmail.com',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Devrajsinh-Jhala/ResearchPlot',
    install_requires=[
        'matplotlib',
        'numpy',
        'seaborn',
        'scikit-learn'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)