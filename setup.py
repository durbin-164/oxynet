import setuptools

with open("README.md", 'r') as rd:
    long_description = rd.read()

with open('requirements.txt', 'r') as rq:
    requirements = [line.strip() for line in rq]

setuptools.setup(
    name='oxynet',
    version='0.0.1',
    author="Md. Masud Rana",
    author_email="masud.cseian@gmail.com",
    description="A deep learning mini framwork",
    long_description=long_description,
    long_description_content_type="text/md",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License  :: OSI approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires='>=3.8',
    install_requires=requirements,

)
