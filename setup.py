import setuptools

setuptools.setup(
    name='quicktorch',
    version='0.0.1',
    author='Anthony DiBenedetto and David Abrutis',
    author_email='adibenedetto117@gmail.com',
    description='Fast and easy wrapper for PyTorch making it easy to use for beginners',
    url='https://github.com/adibenedetto117/quicktorch',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'scikit-learn',
        'matplotlib'
    ]

)