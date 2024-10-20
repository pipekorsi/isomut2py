from setuptools import setup

setup(
    name='isomut2py',
    version='2.0.8',
    description='A mutation detection software from NGS data with multiple filtering and plotting features',
    url='http://github.com/pipekorsi/isomut2',
    author='Orsolya Pipek',
    author_email='pipeko@caesar.elte.hu',
    license='MIT',
    packages=['isomut2py'],
    package_data={'isomut2py': ['alexandrovSignatures/*.csv', 'C/*']},
    install_requires=['matplotlib==3.0.0',
                      'pandas==0.20.3',
                      'theano==1.0.5',
                      'pymc3==3.2',
                      'setuptools==33.1.1',
                      'seaborn==0.8',
                      'scipy==0.19.1',
                      'numpy==1.12.1',
                      'biopython',
                      'joblib==0.11']
)