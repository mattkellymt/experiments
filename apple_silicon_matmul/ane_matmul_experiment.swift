import Foundation
import CoreML
import Accelerate

print("=== Apple Neural Engine (NPU) Matrix Multiply Profiler ===")

// Create CoreML Model Configuration targeting Neural Engine
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine

print("Configured CoreML Compute Units: CPU & Neural Engine (NPU)")
print("CoreML ANE Graph Compilation operates on statically compiled compute graphs.")
