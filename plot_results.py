import csv
import matplotlib.pyplot as plt

grid_sizes = []
cpu_times = []
cpu_frontier_times = []
gpu_frontier_times = []

with open("outputs/results.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows = int(row["rows"])
        cols = int(row["cols"])

        grid_sizes.append(rows * cols)
        cpu_times.append(float(row["cpu_time"]))
        cpu_frontier_times.append(float(row["cpu_frontier_time"]))
        gpu_frontier_times.append(float(row["gpu_frontier_time"]))

plt.figure()
plt.plot(grid_sizes, cpu_times, marker="o", label="CPU BFS")
plt.plot(grid_sizes, cpu_frontier_times, marker="o", label="CPU Frontier BFS")
plt.plot(grid_sizes, gpu_frontier_times, marker="o", label="GPU Frontier BFS")

plt.xlabel("Grid Size (rows × cols)")
plt.ylabel("Runtime (seconds)")
plt.title("BFS Runtime Comparison")
plt.legend()
plt.grid(True)

plt.savefig("outputs/runtime_comparison.png", dpi=300)
plt.show()

gpu_speedup_vs_cpu = []
gpu_speedup_vs_cpu_frontier = []

for cpu, cpu_frontier, gpu in zip(cpu_times, cpu_frontier_times, gpu_frontier_times):
    gpu_speedup_vs_cpu.append(cpu / gpu)
    gpu_speedup_vs_cpu_frontier.append(cpu_frontier / gpu)

plt.figure()
plt.plot(grid_sizes, gpu_speedup_vs_cpu, marker="o", label="GPU vs CPU BFS")
plt.plot(grid_sizes, gpu_speedup_vs_cpu_frontier, marker="o", label="GPU vs CPU Frontier")

plt.xlabel("Grid Size (rows × cols)")
plt.ylabel("Speedup")
plt.title("GPU BFS Speedup")
plt.legend()
plt.grid(True)

plt.savefig("outputs/speedup_comparison.png", dpi=300)
plt.show()
