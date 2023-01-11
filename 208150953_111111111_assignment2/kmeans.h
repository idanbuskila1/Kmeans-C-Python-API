#ifndef KMEANS_H_
#define KMEANS_H_
#include <Python.h>
/*structs def*/
typedef struct cord
{
    double value;
    struct cord *next;
} cord;

typedef struct vector
{
    struct vector *next;
    struct cord *cords;
} vector;

/*prototypes*/
int isConverged(vector *prev_centroids[], vector *centroids[], unsigned K);
int assignDatapointsToCentroid(vector *centroids[], vector *dataPointsByCentroids[], vector *datapoints, unsigned K);
int updateCentroids(vector *centroids[], vector *datapointsByCentroids[], unsigned K, int dimension);
void printCentroids(vector *centroids[], unsigned K);
void freeCords(cord *cor);
void freeVectorGroup(vector *vec, int includeCords);
double d(vector *p, vector *q);
vector **FindKmeans(unsigned K, unsigned N, unsigned iter, float epsilon, unsigned dimension, vector *datapoints, vector **centroids);
vector *ListOfListsToVectors(PyObject *lol, unsigned N, unsigned dimension);
PyObject *VectorArrayToListOfLists(vector **array, unsigned K, unsigned dimension);

#endif