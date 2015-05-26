from setuptools import setup


def my_setup():
    return setup(
        author='Attila Zseder',
        author_email='zseder@gmail.com',
        name='hunvec',
        provides=['hunvec'],
        url='https://github.com/zseder/hunvec',
        packages=[
            'hunvec', 'hunvec.corpus', 'hunvec.nnlm', 'hunvec.utils',
            'hunvec.layers', 'hunvec.seqtag', 'hunvec.feature', 'hunvec.cost',
            'hunvec.datasets'
        ],
        package_dir={'': '.'},
        include_package_data=True,
        zip_safe=False,
        platforms='any',
    )

if __name__ == "__main__":
    my_setup()
