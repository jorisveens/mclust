from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    from numpy.distutils.core import setup, Extension

    mclustaddson = Extension(name='mclust.fortran.mclustaddson',
                             sources=['mclust/fortran/mclustaddson.pyf',
                                      'mclust/fortran/mclustaddson.f'],
                             libraries=['lapack', 'blas'])

    mclust = Extension(name='mclust.fortran.mclust',
                       sources=['mclust/fortran/mclust.pyf',
                                'mclust/fortran/mclust.f',
                                'mclust/fortran/d1mach.f',
                                'mclust/fortran/i1mach.f',
                                ],
                       libraries=['lapack', 'blas'])

    test_files = ['resources/test_data/' + f for f in listdir('resources/test_data') if isfile(join('resources/test_data', f))]

    setup(name='mclust',
          version='0.1',
          packages=['mclust', 'mclust.fortran'],
          ext_modules=[mclustaddson, mclust],
          data_files=[('mclust/resources/data_sets', ['resources/data_sets/diabetes.csv',
                                                      'resources/data_sets/diabetes_classification.csv',
                                                      'resources/data_sets/simulated1d.csv',
                                                      'resources/data_sets/iris.csv']),
                      ('mclust/resources/test_data', test_files)])
