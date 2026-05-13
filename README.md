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
Two files will be created, a bfs_result.png and a results.csv file. An additional file can be used to generate plots in the csv file. By simply running 
`plot_results.py`, it'll output runtime_comparison.png and a speedup_comparison.png plot graphs.
