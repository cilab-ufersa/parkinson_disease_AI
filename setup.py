import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='parkinson_disease_AI',
    url='https://github.com/cilab-ufersa/period_cycle_parkinson_disease_AI',
    author='CILAB',
    author_email='cilab.ufersa@gmail.com',
    # Needed to actually package something
    packages=setuptools.find_packages(),
    include_package_data=True,
    # Needed for dependencies
    install_requires=required,
    description='A package to classify parkinson disease',
    long_description=open('README.md').read(),
)
