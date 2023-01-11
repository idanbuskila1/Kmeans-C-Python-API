from setuptools import Extension, setup

module = Extension("mykmeanssp", sources=[
                   'kmeansmodule.c', 'kmeans.c', 'kmeans.h'])
setup(name='kmeans',
     version='1.0',
     description='Python wrapper for kmeans C extension',
     ext_modules=[module])