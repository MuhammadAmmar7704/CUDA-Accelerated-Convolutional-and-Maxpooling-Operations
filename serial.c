#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_SIZE 300

void readMatrixFromFile(const char* filename, double matrix[MAX_SIZE][MAX_SIZE], int *size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    fscanf(file, "%d", size);

    for (int i = 0; i < *size; i++) {
        for (int j = 0; j < *size; j++) {
            fscanf(file, "%lf", &matrix[i][j]);
        }
    }

    fclose(file);
}

void zeroPad(double matrix[MAX_SIZE][MAX_SIZE], int size, int padWidth, double paddedMatrix[MAX_SIZE][MAX_SIZE]) {
    int paddedSize = size + 2 * padWidth;
    for (int i = 0; i < paddedSize; i++) {
        for (int j = 0; j < paddedSize; j++) {
            if (i < padWidth || j < padWidth || i >= size + padWidth || j >= size + padWidth)
                paddedMatrix[i][j] = 0.0;
            else
                paddedMatrix[i][j] = matrix[i - padWidth][j - padWidth];
        }
    }
}

void print2DMatrix(double matrix[MAX_SIZE][MAX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%lf ",matrix[i][j]);
        }            
            printf("\n");
    }
}


void convolve2D(double input[MAX_SIZE][MAX_SIZE], double kernel[MAX_SIZE][MAX_SIZE], 
                double output[MAX_SIZE][MAX_SIZE], int inputSize, int kernelSize) {
    //printf("input :\n" );
    //print2DMatrix(input,inputSize,inputSize);
    //printf("\nkernel :\n" );
    //print2DMatrix(kernel,kernelSize,kernelSize);
    
    for (int y = 0; y < inputSize - kernelSize + 1; y++) {
        for (int x = 0; x < inputSize - kernelSize + 1; x++) {
            double sum = 0.0;
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    sum += input[y + i][x + j] * kernel[i][j];
                }
            }
            //printf("sum = %lf, y = %d, x = %d", sum, y, x);
            output[y][x] = sum;
        }
    }
    //printf("\noutput :\n" );
    //print2DMatrix(output,inputSize - kernelSize + 1,inputSize - kernelSize + 1 );
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void applySigmoid(double matrix[MAX_SIZE][MAX_SIZE], int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = sigmoid(matrix[i][j]);
        }
    }
}

void maxPooling(double input[MAX_SIZE][MAX_SIZE], double output[MAX_SIZE][MAX_SIZE],
                int inputSize, int poolSize, int stride) {
    int outputSize = (inputSize - poolSize) / stride + 1;
    for (int y = 0; y < outputSize; y++) {
        for (int x = 0; x < outputSize; x++) {
            double maxVal = -INFINITY;
            for (int i = 0; i < poolSize; i++) {
                for (int j = 0; j < poolSize; j++) {
                    double val = input[y * stride + i][x * stride + j];
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
            output[y][x] = maxVal;
        }
    }
    //printf("\n\nAfter Pooling = \n");
    //print2DMatrix(output,outputSize ,outputSize);
    //printf("\n");
}


    double input[MAX_SIZE][MAX_SIZE];
    double kernel1[MAX_SIZE][MAX_SIZE], kernel2[MAX_SIZE][MAX_SIZE], kernel3[MAX_SIZE][MAX_SIZE];
int main() {
    int inputSize, kernelSize;
    
    double startTime = omp_get_wtime();
    
    readMatrixFromFile("input.txt", input, &inputSize);
    readMatrixFromFile("kernel1.txt", kernel1, &kernelSize);
    readMatrixFromFile("kernel2.txt", kernel2, &kernelSize);
    readMatrixFromFile("kernel3.txt", kernel3, &kernelSize);

    double paddedInput[MAX_SIZE][MAX_SIZE];
    zeroPad(input, inputSize, kernelSize / 2, paddedInput);
    int paddedSize = inputSize + kernelSize - 1;


    
    
    double conv1[MAX_SIZE][MAX_SIZE], conv2[MAX_SIZE][MAX_SIZE], conv3[MAX_SIZE][MAX_SIZE];
    convolve2D(paddedInput, kernel1, conv1, paddedSize, kernelSize);
    convolve2D(paddedInput, kernel2, conv2, paddedSize, kernelSize);
    convolve2D(paddedInput, kernel3, conv3, paddedSize, kernelSize);

    //printf("\n\n\nconv1 after convolve2D : \n");
    //print2DMatrix(conv1,inputSize ,inputSize);
    //printf("\n\n\n");
    
    
    applySigmoid(conv1, paddedSize);
    applySigmoid(conv2, paddedSize);
    applySigmoid(conv3, paddedSize);
    
    //printf("\n\n\nconv1 after sigmoid : \n");
   // print2DMatrix(conv1,inputSize ,inputSize);
    //printf("\n\n\n");

    double pooled1[MAX_SIZE][MAX_SIZE], pooled2[MAX_SIZE][MAX_SIZE], pooled3[MAX_SIZE][MAX_SIZE];
    maxPooling(conv1, pooled1, paddedSize, 2, 2);
    maxPooling(conv2, pooled2, paddedSize, 2, 2);
    maxPooling(conv3, pooled3, paddedSize, 2, 2);

    FILE *file = fopen("outputserial.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
  int poolSize = 2, stride = 2;
  //int paddedInputSize = inputSize + 2 * padding; // If padding is applied
  int outputSize = (paddedSize - poolSize) / stride + 1;


    fprintf(file, "Max Pooled Result:\n[");
	for (int i = 0; i < outputSize; i++) {
	    fprintf(file, "[");
	    for (int j = 0; j < outputSize; j++) {
		fprintf(file, "%.8e", pooled1[i][j]);
		if (j < outputSize - 1) fprintf(file, " ");
	    }
	    fprintf(file, "]");
	    if (i < outputSize - 1) fprintf(file, "\n ");
	}
	fprintf(file, "]\n\n[");

	for (int i = 0; i < outputSize; i++) {
	    fprintf(file, "[");
	    for (int j = 0; j < outputSize; j++) {
		fprintf(file, "%.8e", pooled2[i][j]);
		if (j < outputSize - 1) fprintf(file, " ");
	    }
	    fprintf(file, "]");
	    if (i < outputSize - 1) fprintf(file, "\n ");
	}
	fprintf(file, "]\n\n[");

	for (int i = 0; i < outputSize; i++) {
	    fprintf(file, "[");
	    for (int j = 0; j < outputSize; j++) {
		fprintf(file, "%.8e", pooled3[i][j]);
		if (j < outputSize - 1) fprintf(file, " ");
	    }
	    fprintf(file, "]");
	    if (i < outputSize - 1) fprintf(file, "\n ");
	}
	fprintf(file, "]]\n");
	
    double endTime = omp_get_wtime();
    double totalTime = endTime - startTime;

    
    printf("Total execution time: %.10f seconds\n", totalTime);
    
    
    fclose(file);
    return 0;
}

