import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="mtrec",
  version="0.0.1",
  author="Ziyao Geng",
  author_email="zggzy1996@163.com",
  description="A simple package about multi-task recommendation",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/ZiyaoGeng/MTRec",
  packages=setuptools.find_packages(),
  python_requires=">=3.6",
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  license="Apache-2.0",
)