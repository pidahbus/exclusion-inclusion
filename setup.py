import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='exclusion-inclusion',
    version='0.1',
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
    packages=['exclusion-inclusion'],
    install_requires=['numpy', 'pandas', 'tensorflow'],
)
