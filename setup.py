import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Neuro",
    version="0.1",
    author="Mohamed Khalil",
    author_email="mohammed.khalil@mail.utoronto.ca",
    description="A small pedagogical machine learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/momokhalil/Neuro",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy']
)
