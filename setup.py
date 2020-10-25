from setuptools import setup, find_packages

version = "0.1.1"

setup(
    name="aristote",
    version=version,
    license="proprietary",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
