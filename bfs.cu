#include <stdio.h>
#include <stdbool.h>
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
// #include <opencv4>

using namespace std;

// #define MAX 100 //max grid size
#define MAX 1024
#define ROWS 256
#define COLS 256
// #define ROWS 1024
// #define COLS 1024
#define MAX_CELLS (ROWS * COLS)

#define CHECK(call){ \
    const cudaError_t cuda_ret = call;\
    if (cuda_ret != cudaSuccess)  {\
        printf("error: %s:%d, ", __FILE__, __LINE__);\
        printf("code: %d, reason:%s\n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1);\
    }\
}\

double myCPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) tp.tv_sec + (double)tp.tv_usec/1.0e6;
}

// typedef struct{
//     int data[MAX_CELLS];
//     int front;
//     int rear;
// } Queue;

typedef struct{
    int* data;
    int front;
    int rear;
} Queue;

void initQueue(Queue* q, int maxSize){
    // q->front = 0;
    // q->rear = 0;

    q->data = (int*)malloc(maxSize * sizeof(int));
    q->front = 0;
    q->rear = 0;
}

bool isEmpty(Queue* q){
    return q->front == q->rear;
}

void enqueue(Queue* q, int value){
    q->data[q->rear++] = value;
}

int dequeue(Queue* q){
    return q->data[q->front++];
}

bool isValid(int r, int c, int rows, int cols){
    return r >= 0 && r < rows && c >= 0 &&  c < cols;
}

int getIndex(int r, int c, int cols){
    return r * cols + c;
}

void getRowCol(int idx, int cols, int* r, int* c){
    *r = idx / cols;
    *c = idx % cols;
}

__global__ void expand_frontier(int* grid, int* visited, int* frontier, int frontierSize, int* nextFrontier, int* nextFrontierSize, int* parent, int rows, int cols, int goalIdx, int* goalFound){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSize) return;

    int current = frontier[tid];
    int row = current / cols;
    int col = current % cols;

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    for (int d = 0; d < 4; d++){
        int nr = row + dr[d];
        int nc = col + dc[d];

        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols){
            int neighbor = nr * cols + nc;

            if (grid[neighbor] == 0){
                if (atomicExch(&visited[neighbor], 1) == 0){
                    parent[neighbor] = current;
                    int pos = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[pos] = neighbor;

                    if(neighbor == goalIdx){
                        // *goalFound = 1;
                        atomicExch(goalFound, 1);
                    }
                }
            }
        }
    }
}

bool bfs_frontier_gpu(int* grid, int rows, int cols, int startIdx, int goalIdx, int* parent, float* gpuTime){

    int size = rows * cols;

    int *d_grid, *d_visited, *d_frontier, *d_nextFrontier;
    int *d_nextFrontierSize, *d_parent, *d_goalFound;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeMalloc = 0;
    float timeMemcpy = 0;
    float time1 = 0;
    float timeFrontier = 0;

    cudaEventRecord(start);
    CHECK(cudaMalloc(&d_grid, size * sizeof(int)));
    CHECK(cudaMalloc(&d_visited, size * sizeof(int)));
    CHECK(cudaMalloc(&d_frontier, size * sizeof(int)));
    CHECK(cudaMalloc(&d_nextFrontier, size * sizeof(int)));
    CHECK(cudaMalloc(&d_nextFrontierSize, sizeof(int)));
    CHECK(cudaMalloc(&d_parent, size * sizeof(int)));
    CHECK(cudaMalloc(&d_goalFound, sizeof(int)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    timeMalloc += time1;

    int* visited = (int*)calloc(size, sizeof(int));

    cudaEventRecord(start);
    CHECK(cudaMemcpy(d_grid, grid, size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_visited, visited, size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_parent, parent, size * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    time1 = 0;
    cudaEventElapsedTime(&time1, start, stop);
    timeMemcpy += time1;

    // int frontier[MAX_CELLS];
    // int nextFrontier[MAX_CELLS];
    int* frontier = (int*)malloc(size * sizeof(int));
    int* nextFrontier = (int*)malloc(size * sizeof(int));

    int frontierSize = 1;
    frontier[0] = startIdx;

    visited[startIdx] = 1;
    parent[startIdx] = -1;

    cudaEventRecord(start);
    CHECK(cudaMemcpy(d_frontier, frontier, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_visited + startIdx, &visited[startIdx], sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_parent + startIdx, &parent[startIdx], sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    time1 = 0;
    cudaEventElapsedTime(&time1, start, stop);
    timeMemcpy += time1;

    int goalFound = 0;

    while (frontierSize > 0 && ! goalFound){
        int zero = 0;

        cudaEventRecord(start);
        CHECK(cudaMemcpy(d_nextFrontierSize, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_goalFound, &zero, sizeof(int), cudaMemcpyHostToDevice));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        time1 = 0;
        cudaEventElapsedTime(&time1, start, stop);
        timeMemcpy += time1;

        int threads = 256;
        int blocks = (frontierSize + threads - 1) / threads;

        cudaEventRecord(start);
        expand_frontier<<<blocks, threads>>>(d_grid, d_visited, d_frontier, frontierSize, d_nextFrontier, d_nextFrontierSize, d_parent, rows, cols, goalIdx, d_goalFound);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        time1 = 0;
        cudaEventElapsedTime(&time1, start, stop);
        timeFrontier += time1;

        cudaDeviceSynchronize();

        int nextSize;
        cudaEventRecord(start);
        CHECK(cudaMemcpy(&nextSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&goalFound, d_goalFound, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(nextFrontier, d_nextFrontier, nextSize * sizeof(int), cudaMemcpyDeviceToHost));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        time1 = 0;
        cudaEventElapsedTime(&time1, start, stop);
        timeMemcpy += time1;

        //swap
        for (int i = 0; i < nextSize; i++){
            frontier[i] = nextFrontier[i];
        }

        frontierSize = nextSize;
        
        cudaEventRecord(start);
        CHECK(cudaMemcpy(d_frontier, frontier, frontierSize * sizeof(int), cudaMemcpyHostToDevice));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        time1 = 0;
        cudaEventElapsedTime(&time1, start, stop);
        timeMemcpy += time1;
    }

    // cudaMemcpy(d_frontier, frontier, frontierSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    CHECK(cudaMemcpy(parent, d_parent, size*sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    time1 = 0;
    cudaEventElapsedTime(&time1, start, stop);
    timeMemcpy += time1;

    cudaFree(d_grid);
    cudaFree(d_visited);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextFrontierSize);
    cudaFree(d_parent);
    cudaFree(d_goalFound);
    free(frontier);
    free(nextFrontier);

    printf("Kernel: cudamemcpy: %f s\n", timeMemcpy / 1000.0f);
    printf("Kernel: cudaMalloc: %f s\n", timeMalloc / 1000.0f);
    printf("expand_frontier: %f s\n", timeFrontier / 1000.0f);
    printf("GPU on bfs_frontier_gpu: %f s\n", (timeMemcpy + timeMalloc + timeFrontier) / 1000.0f);
    *gpuTime = ((timeMemcpy + timeMalloc + timeFrontier) / 1000.0f);

    return goalFound;
}

bool bfs_frontier_cpu(int* grid, int rows, int cols, int startIdx, int goalIdx, int* parent, int* visited){
    // int frontier[MAX_CELLS];
    // int nextFrontier[MAX_CELLS];

    int size = rows * cols;

    int* frontier = (int*)malloc(size * sizeof(int));
    int* nextFrontier = (int*)malloc(size * sizeof(int));

    int frontierSize = 0;
    int nextFrontierSize = 0;

    frontier[0] = startIdx;
    frontierSize = 1;
    visited[startIdx] = 1;
    parent[startIdx] = -1;

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    while (frontierSize > 0){
        nextFrontierSize = 0;
        for (int i = 0; i < frontierSize; i++){
            int current = frontier[i];

            if (current == goalIdx){
                free(frontier);
                free(nextFrontier);
                return true;
            }

            int row, col;
            getRowCol(current, cols, &row, &col);

            for (int d = 0; d < 4; d++){
                int nr = row + dr[d];
                int nc = col + dc[d];

                if (isValid(nr, nc, rows, cols)){
                    int neighbor = getIndex(nr, nc, cols);

                    if (grid[neighbor] == 0 && visited[neighbor] == 0){
                        visited[neighbor] = 1; 
                        parent[neighbor] = current;
                        nextFrontier[nextFrontierSize++] = neighbor;
                    }
                }
            }
        }

        for (int i = 0; i < nextFrontierSize; i++){
            frontier[i] = nextFrontier[i];
        }
        frontierSize = nextFrontierSize;
    }
    free(frontier);
    free(nextFrontier);
    return false;
}

bool bfs(int* grid, int rows, int cols, int startIdx, int goalIdx, bool* visited, int* parent){
    Queue q;
    initQueue(&q, rows * cols);

    enqueue(&q, startIdx);
    visited[startIdx] = true;
    parent[startIdx] = -1;

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    while (!isEmpty(&q)){
        int currentIdx = dequeue(&q);

        if (currentIdx == goalIdx){
            free(q.data);
            return true;
        }

        int row, col;
        getRowCol(currentIdx, cols, &row, &col);

        for (int i = 0; i < 4; i++){
            int nr = row + dr[i];
            int nc = col + dc[i];

            if (isValid(nr, nc, rows, cols)){
                int neighborIdx = getIndex(nr, nc, cols);

                if (!visited[neighborIdx] && grid[neighborIdx] == 0){
                    visited[neighborIdx] = true;
                    parent[neighborIdx] = currentIdx;
                    enqueue(&q, neighborIdx);
                }
            }
        }
    }

    free(q.data);

    return false;
}

void reconstructPath(int* parent, int startIdx, int goalIdx, int cols){
    // Point path[MAX * MAX];
    // int length = 0;

    // Point curr = goal;

    // while (!(curr.rows == -1 && curr.cols == -1)){
    //     path[length++] = curr;
    //     curr = parent[curr.rows][curr.cols];
    // }
    // printf("PATH (reversed): \n");
    // for (int i = length - 1; i >= 0; i--){
    //     printf("(%d, %d) ", path[i].rows, path[i].cols);
    // }
    // printf("\n Path length: %d\n", length - 1);
    int path[MAX_CELLS];
    int length = 0;

    int curr = goalIdx;
    while (curr != -1){
        path[length++] = curr;
        curr = parent[curr];
    }

    printf("Path:\n");
    for (int i = length -1; i >= 0; i--){
        int r, c;
        getRowCol(path[i], cols, &r, &c);
        printf("(%d, %d) ", r, c);
    }
    printf("\nPath length: %d\n", length - 1);
}

void printGrid(int* grid, int rows, int cols, int startIdx, int goalIdx) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = getIndex(r, c, cols);

            if (idx == startIdx) {
                printf("S ");
            } else if (idx == goalIdx) {
                printf("G ");
            } else if (grid[idx] == 1) {
                printf("# ");
            } else {
                printf(". ");
            }
        }
        printf("\n");
    }
}

void saveGridPPM(const char* filename, int* grid, bool* visited, int* parent, int rows, int cols, int startIdx, int goalIdx) {
    FILE* fp = fopen(filename, "w");
    fprintf(fp, "P3\n%d %d\n255\n", cols, rows);

    bool* path = (bool*)calloc(rows * cols, sizeof(bool));

    int cur = goalIdx;
    while (cur != -1 && cur != startIdx) {
        path[cur] = true;
        cur = parent[cur];
    }
    path[startIdx] = true;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;

            if (idx == startIdx)
                fprintf(fp, "0 255 0 ");
            else if (idx == goalIdx)
                fprintf(fp, "255 0 0 ");
            else if (path[idx])
                fprintf(fp, "0 0 255 ");
            else if (grid[idx] == 1)
                fprintf(fp, "0 0 0 ");
            else if (visited[idx])
                fprintf(fp, "180 180 180 ");
            else
                fprintf(fp, "255 255 255 ");
        }
        fprintf(fp, "\n");
    }

    free(path);
    fclose(fp);
}

int main(){
    mkdir("outputs", 0777);
    // int grid[MAX_CELLS] = {
    //     0, 0, 0, 0, 1, 0,
    //     1, 1, 0, 0, 1, 0,
    //     0, 0, 0, 1, 0, 0,
    //     0, 1, 1, 0, 0, 1,
    //     0, 0, 0, 0, 0, 0
    // };

    // int grid[MAX_CELLS];
    int* grid = (int*)malloc(MAX_CELLS * sizeof(int));
    bool* visited = (bool*)calloc(MAX_CELLS, sizeof(bool));
    int* parent = (int*)malloc(MAX_CELLS * sizeof(int));
    int* visited1 = (int*)calloc(MAX_CELLS, sizeof(int));
    int* parent1 = (int*)malloc(MAX_CELLS * sizeof(int));

    srand(time(NULL));

    for (int r = 0; r < ROWS; r++){
        for (int c = 0; c < COLS; c++){
            grid[r*COLS + c] = (rand() % 100 < 30) ? 1 : 0;
        }
    }

    // bool visited[MAX_CELLS] = {false};
    // int parent[MAX_CELLS];
    // int visited1[MAX_CELLS] = {-1};

    for (int i = 0; i < MAX_CELLS; i++){
        parent[i] = -1;
        parent1[i] = -1;
    }

    int startIdx = getIndex(0, 0, COLS);
    // int goalIdx = getIndex(4, 5, COLS);
    int goalIdx = getIndex(ROWS - 1, COLS - 1, COLS);

    if(grid[startIdx] == 1 || grid[goalIdx] == 1){
        printf("Start or goal blocked.\n");
        return 1;
    }

    // printGrid(grid, ROWS, COLS, startIdx, goalIdx);

    double start = myCPUTimer();
    bool found = bfs(grid, ROWS, COLS, startIdx, goalIdx, visited, parent);
    double end = myCPUTimer();
    double cpuTime = end - start;
    printf("CPU TIME: %f seconds\n", end - start);

    start = myCPUTimer();
    bool found1 = bfs_frontier_cpu(grid, ROWS, COLS, startIdx, goalIdx, parent1, visited1);
    end = myCPUTimer();
    double cpuFrontierTime = end - start;
    printf("CPU FRONTIER TIME: %f seconds\n", end - start);

    if (found) {
        printf("Path found!\n");
        // reconstructPath(parent, startIdx, goalIdx, COLS);
    } else {
        printf("No path found.\n");
    }

    //---------------------------------------------------------------------------
    int parent_gpu[MAX_CELLS];

    for (int i = 0; i < MAX_CELLS; i++){
        parent_gpu[i] = -1;
    }

    float gpuFrontierTime = 0; 
    bool found_gpu = bfs_frontier_gpu(grid, ROWS, COLS, startIdx, goalIdx, parent_gpu, &gpuFrontierTime);
    printf("GPU frontier BFS result: %s\n", found_gpu ? "Path found" : "No path found");

    if (found == found_gpu && found1 == found_gpu){
        printf("CPU and GPU are the same.\n");
    } else {
        printf("Mismatch: CPU and GPU disagree.\n");
    }

    if (found){
        saveGridPPM("outputs/bfs_result.ppm", grid, visited, parent, ROWS, COLS, startIdx, goalIdx);
    
        FILE* fp = fopen("outputs/results.csv", "a");
        // fprintf(fp, "rows,cols,cpu_time,cpu_frontier_time,gpu_frontier_time\n");
        fprintf(fp, "%d,%d,%f,%f,%f\n",
            ROWS,
            COLS,
            cpuTime,
            cpuFrontierTime,
            gpuFrontierTime);
        fclose(fp);
    }

    free(grid);
    free(visited);
    free(parent);
    free(visited1);
    free(parent1);
    return 0;

}
// module spider opencv
// nvcc -O2 bfs.cu -o bfs_cpu
