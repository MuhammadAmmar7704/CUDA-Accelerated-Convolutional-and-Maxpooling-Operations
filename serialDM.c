#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>  // For sleep()

#define MAX_SIZE 5000

// Define a global variable to signal when computation ends
int computationFinished = 0;

// Timer function to print a message every 20 seconds
void* timerThread(void* arg) {
    while (!computationFinished) {
        sleep(2);  // Sleep for 20 seconds
        if (!computationFinished) {
            printf("1 minute have passed...\n");
        }
    }
    pthread_exit(NULL);
}

double** allocate2DArray(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        array[i] = (double*)malloc(cols * sizeof(double));
    }
    return array;
}

void free2DArray(double** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void readMatrixFromFile(const char* filename, double** matrix, int* size) {
    FILE* file = fopen(filename, "r");
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

void zeroPad(double** matrix, int size, int padWidth, double** paddedMatrix, int paddedSize) {
    //int paddedSize = size + 2 * padWidth;
    //int paddedSize = size + padWidth;
    
    //printf("here = %d\n", paddedSize);
    for (int i = 0; i < paddedSize; i++) {
        for (int j = 0; j < paddedSize; j++) {
            if (i < padWidth || j < padWidth || i >= size + padWidth || j >= size + padWidth)
                paddedMatrix[i][j] = 0.0;
            else
                paddedMatrix[i][j] = matrix[i - padWidth][j - padWidth];
        }
    }
}

void convolve2D(double** input, double** kernel, double** output, int inputSize, int kernelSize) {
    for (int y = 0; y < inputSize - kernelSize + 1; y++) {
        for (int x = 0; x < inputSize - kernelSize + 1; x++) {
            double sum = 0.0;
            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    sum += input[y + i][x + j] * kernel[i][j];
                }
            }
            output[y][x] = sum;
        }
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void applySigmoid(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = sigmoid(matrix[i][j]);
        }
    }
}

void maxPooling(double** input, double** output, int inputSize, int poolSize, int stride) {
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
}

void print2DMatrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%lf ",matrix[i][j]);
        }            
            printf("\n");
    }
}

int main() {

    pthread_t timerThreadID;

    // Start the timer thread
    pthread_create(&timerThreadID, NULL, timerThread, NULL);

    int inputSize, kernelSize;

    //printf("reached here");
    // Allocate arrays dynamically
    double** input = allocate2DArray(MAX_SIZE, MAX_SIZE);
    double** kernel1 = allocate2DArray(MAX_SIZE, MAX_SIZE);
    double** kernel2 = allocate2DArray(MAX_SIZE, MAX_SIZE);
    double** kernel3 = allocate2DArray(MAX_SIZE, MAX_SIZE);
    

    double startTime = omp_get_wtime();
    
    readMatrixFromFile("input.txt", input, &inputSize);
    readMatrixFromFile("kernel1.txt", kernel1, &kernelSize);
    readMatrixFromFile("kernel2.txt", kernel2, &kernelSize);
    readMatrixFromFile("kernel3.txt", kernel3, &kernelSize);
    
    //print2DMatrix(input,inputSize,inputSize);
    
    //int paddedSize = inputSize + kernelSize - 1;
    int paddedSize = inputSize + (kernelSize / 2) * 2;
        
    double** paddedInput = allocate2DArray(paddedSize, paddedSize);
    zeroPad(input, inputSize, kernelSize / 2, paddedInput,paddedSize);
    
    double** conv1 = allocate2DArray(paddedSize, paddedSize);
    double** conv2 = allocate2DArray(paddedSize, paddedSize);
    double** conv3 = allocate2DArray(paddedSize, paddedSize);

    convolve2D(paddedInput, kernel1, conv1, paddedSize, kernelSize);
    convolve2D(paddedInput, kernel2, conv2, paddedSize, kernelSize);
    convolve2D(paddedInput, kernel3, conv3, paddedSize, kernelSize);

    applySigmoid(conv1, paddedSize);
    applySigmoid(conv2, paddedSize);
    applySigmoid(conv3, paddedSize);

    int poolSize = 2, stride = 2;
    int outputSize = (paddedSize - poolSize) / stride + 1;
    double** pooled1 = allocate2DArray(outputSize, outputSize);
    double** pooled2 = allocate2DArray(outputSize, outputSize);
    double** pooled3 = allocate2DArray(outputSize, outputSize);

    maxPooling(conv1, pooled1, paddedSize, poolSize, stride);
    maxPooling(conv2, pooled2, paddedSize, poolSize, stride);
    maxPooling(conv3, pooled3, paddedSize, poolSize, stride);
    
    
    
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
  //int poolSize = 2, stride = 2;
  //int paddedInputSize = inputSize + 2 * padding; // If padding is applied
  //int outputSize = (paddedSize - poolSize) / stride + 1;


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
	
    //double endTime = omp_get_wtime();
    //double totalTime = endTime - startTime;

    
    //printf("Total execution time: %.2f seconds\n", totalTime);
    
    
    fclose(file);

    
    
    free2DArray(paddedInput, paddedSize);
    free2DArray(conv1, paddedSize);
    free2DArray(conv2, paddedSize);
    free2DArray(conv3, paddedSize);
    free2DArray(pooled1, outputSize);
    free2DArray(pooled2, outputSize);
    free2DArray(pooled3, outputSize);
    /*
    */
    double endTime = omp_get_wtime();
    printf("Total execution time: %.2f seconds\n", endTime - startTime);
    
    // Free allocated memory
    free2DArray(input, MAX_SIZE);
    free2DArray(kernel1, MAX_SIZE);
    free2DArray(kernel2, MAX_SIZE);
    free2DArray(kernel3, MAX_SIZE);
    
    // Signal the timer thread to stop and join it
    computationFinished = 1;
    pthread_join(timerThreadID, NULL);

    return 0;
}


