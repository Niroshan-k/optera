from setuptools import setup, find_packages

setup(
    name="optera",
    version="1.0.0",
    packages=find_packages(include=["optera", "optera.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scipy>=1.8.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "reportlab>=3.6.0"
    ],
    author="Optera Core Development Team",
    description="Quantitative Supply Chain Framework & Stochastic Portfolio Optimization Engine",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Niroshan-k/Event-Driven-Simulation-Framework",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
)
