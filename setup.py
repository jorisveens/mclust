if __name__ == '__main__':
    from numpy.distutils.core import setup, Extension

    mclustaddson = Extension(name='mclust.fortran.mclustaddson',
                        sources=['mclust/fortran/mclustaddson.pyf',
                                 'mclust/fortran/mclustaddson.f'],
                        libraries=['lapack', 'blas'])

    mclust = Extension(name='mclust.fortran.mclust',
                       sources=['mclust/fortran/mclust.pyf',
                                'mclust/fortran/mclust.f',
                                'mclust/fortran/mclustaddson.f'],
                       libraries=['lapack', 'blas', 'slatec', 'gfortran', 'm', 'quadmath', 'R'])

    setup(name='mclust',
          version=0.0,
          packages=['mclust', 'mclust.fortran'],
          ext_modules=[mclustaddson, mclust])
