# GPU_BFS_Frontier
The main file is called **bfs.cu**. In order to run this file, it is recommended that you have a really nice computer but in my case I used the HPC at NCSA. Below are the instructions to compile the file

If in the GPU Node, make sure to load opencv module. 

`module load opencv/4.13.0.x86_64`

To compile, do the following : 

`nvcc -o bfs bfs.cu -I $OPENCV_HOME/include/opencv4/ -L $OPENCV_HOME/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc`

Simply run it: `./bfs`

In order to change grid dimensions, go to lines 19 and 20 and enter the desired dimensions. 

`#define ROWS 8192`

`#define COLS 8192`

___
Two files will be created, a bfs_result.ppm and a results.csv file. **.ppm** files should be converted to png files. An additional file can be used to generate plots in the csv file. By simply running 
`python3 plot_results.py`, it'll output runtime_comparison.png and a speedup_comparison.png plot graphs.

It is necesary to make a directory called **outputs** and in that directory, make a file called `results.csv`. In that file on line 1, enter `rows,cols,cpu_time,cpu_frontier_time,gpu_frontier_time` and press enter, that way cursor is at the line two that way when you run the file, it'll append all new results to the second line and on.

All outputs inluding .png, .ppm, .csv files will be made/modified in **outputs** directory
