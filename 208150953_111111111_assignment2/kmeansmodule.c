#include <stdio.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include "kmeans.h"
#include <math.h>

/*gets python object pointer to list of N lists of size dimension.
returns N connected vectors with |dimension| cords each, which represents the list of lists
if an error occured return NULL */
vector *ListOfListsToVectors(PyObject *lol, unsigned N, unsigned dimension)
{
    PyObject *PyPoint;
    PyObject *item;
    vector *curVec=NULL, *headVec = NULL, *tmpVec;
    cord *headCord = NULL, *curCord;
    unsigned i, j;
    double val;

    for (i = 0; i < N; i++)
    {
        tmpVec = (vector *)malloc(sizeof(vector *));
        curCord = (cord *)malloc(sizeof(cord *));
        if (tmpVec == NULL || curCord==NULL){/*if memory error occured*/
            freeVectorGroup(headVec, 1);/*free vectors already malloced(if exist)*/
            if(tmpVec!=NULL) free(tmpVec);
            if(curCord!=NULL) free(curCord);
            return NULL;
        }
        PyPoint = PyList_GetItem(lol, i);
        for (j = 0; j < dimension; j++)
        {
            item = PyList_GetItem(PyPoint, j);
            val = PyFloat_AsDouble(item);
            curCord->value = val;
            if (j == 0) headCord = curCord; /* first cord in vector */
            if (j != dimension - 1)
            {
                curCord->next = (cord *)malloc(sizeof(cord *));
                curCord = curCord->next;
                if (curCord == NULL){ /*if memory error occured */
                    freeCords(headCord);/*free cords already malloced(if exist)*/
                    return NULL;
                }
            }
            else curCord->next = NULL;/* last cord in vector */
        }
        tmpVec->cords = headCord;
        tmpVec->next = NULL;
        if (i == 0) /* first vector in list */
        {
            headVec = tmpVec;
            curVec = headVec;
        }
        else
        {
            curVec->next = tmpVec;
            curVec = curVec->next;
        }
    }
    return headVec;
}
/*gets array with K pointers to vectors with |dimension| cords.
returns python object pointer to a python list of K lists of size |dimension|"*/
PyObject *VectorArrayToListOfLists(vector **array, unsigned K, unsigned dimension){
    PyObject *retList = PyList_New(K);
    unsigned i=0,j=0;
    PyObject* python_float;
    cord *tmp;

    for(;i<K;i++){
        PyObject *innerList = PyList_New(dimension);
        tmp = array[i]->cords;
        for(;j<dimension;j++){
            python_float = Py_BuildValue("f", tmp->value);
            PyList_SetItem(innerList, j, python_float);
            tmp = tmp->next;
        }
        PyList_SetItem(retList, i, innerList);
        j=0;
    }
    return retList;
}
static PyObject* fit(PyObject *self, PyObject *args)
{
    unsigned K, N, iter, dimension, i=0;
    float epsilon;
    PyObject *dataset;
    PyObject *PyCentroids, *retList;
    vector *datapoints, **centroids, *tmpVec, *tmpCentroids;

    /*parse variables from python*/
    if (!PyArg_ParseTuple(args, "IIIIfOO", &K, &N, &iter, &dimension, &epsilon, &dataset, &PyCentroids))
        return NULL;
    centroids = (vector **)malloc(sizeof(vector *) * K); /* array of K pointers*/
    tmpCentroids = ListOfListsToVectors(PyCentroids, K, dimension);
    datapoints = ListOfListsToVectors(dataset, N, dimension);
    tmpVec = tmpCentroids;
    if (datapoints != NULL && tmpCentroids != NULL && centroids!= NULL){/* all mallocs were succesful: */
        for (; i < K; i++)
        { /*store centroids adresses in the designated pointers array */
            centroids[i] = tmpVec;
            tmpVec = tmpVec->next;
        }
        /*find k means with initial centroids received from kmeans_pp.py*/
        centroids = FindKmeans(K, N, iter, epsilon, dimension, datapoints, centroids);
        if (centroids != NULL)
        {
            /*build PyObject list of lists of the calculated centroids*/
            retList = VectorArrayToListOfLists(centroids, K, dimension);
            /* free all memory */
            i=0;
            for (; i < K; i++){
                if (centroids[i] != NULL)
                {
                    if (centroids[i]->cords != NULL)
                        freeCords(centroids[i]->cords);
                    free(centroids[i]);
                }
            }
            free(centroids);
            return retList;
        }
    }
    if(centroids!=NULL) free(centroids);
    PyErr_SetString(PyExc_RuntimeError, "An error has occured");
    return NULL;
    
    
}
static PyMethodDef KmeansMethods[] = {
    {"fit",                                                            /* the Python method name that will be used */
     (PyCFunction)fit,                                                 /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                                                     /* flags indicating parameters accepted for this function */
     PyDoc_STR("Find K-means for d-dimensional datapoints dataset arguments:K- num of centroids. N- num of datapoints. iter- max iteration num. dimension- datapoint dimension. epsilon- accuracy level. datapoints- datapoints to find kmeans for centroids- initial centroids for the algorithm")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}   /* The last entry must be all NULL as shown to act as a
                            sentinel. Python looks for this entry to know that all
                            of the functions for the module have been defined. */
};
static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",                                                            /* name of module */
    "implements kmeans algorithm on dataset of N d-dimensional datapoints ", /* module documentation, may be NULL */
    -1,                                                                      /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    KmeansMethods                                                            /* the PyMethodDef array from before containing the methods of the extension */
};
PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&kmeansmodule);
    if (!m)
    {
        return NULL;
    }
    return m;
}