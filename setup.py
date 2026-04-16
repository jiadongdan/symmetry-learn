from setuptools import setup, find_packages

setup(
    name='symmetry-learn',
    packages=find_packages(),
    version='0.1',
    license='MIT',
    description='Deep learning for STEM images by symmetry',
    author='Jiadong Dan',
    author_email='',
    url='',
    keywords=['symmetry', 'deep learning'],
    install_requires=[
        'numpy',
        'scipy',
        'ase',
        'scikit-image',
        'scikit-learn',
        'spglib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)
