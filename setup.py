import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ei',
    version='0.1.2',
    author='Subhadip Maji',
    author_email='subhadipmaji.jumech@gmail.com',
    description=('A model agnostic approach to calculate feature importance with direction'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.optum.com/DataScienceIndia/exclusion-inclusion',
    project_urls={
        "Bug Tracker": "https://github.optum.com/DataScienceIndia/exclusion-inclusion/issues"
    },
    license='',
    packages=['ei'],
    install_requires=['numpy>=1.19.5', 'pandas>=1.2.4', 'tensorflow>=2.5.0', 'loguru>=0.3.2'],
)

