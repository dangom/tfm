from io import open

from setuptools import find_packages, setup

with open('tfm/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.org', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = ['nibabel', 'numpy', 'sklearn',
            'seaborn', 'matplotlib', 'pandas']

setup(
    name='tfm',
    version=version,
    description='Compute temporal functional modes of activation',
    long_description=readme,
    author='Daniel Gomez',
    author_email='d.gomez@donders.ru.nl',
    maintainer='Daniel Gomez',
    maintainer_email='d.gomez@donders.ru.nl',
    url='https://github.com/dangom/tfm',
    license='GNU GPL-3.0',

    keywords=[
        'temporal functional modes',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPL3 License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    entry_points={
        'console_scripts': [
            'tfm = tfm.tfm:run_tfm',
            'tfm_confounds=tfm.tfm:run_correlation_with_confounds',
            'raicar = tfm.raicar:main'
        ]
    },
    install_requires=REQUIRES,
    tests_require=['coverage', 'pytest'],

    packages=find_packages(),
)
