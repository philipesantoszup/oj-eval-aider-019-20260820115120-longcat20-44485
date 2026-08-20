#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t m = i + 1;
    
    // Move all keys and values to SRAM
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);
    }
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Create K_all by concatenating keys
    Matrix* K_all = matrix_memory_allocator.Allocate("K_all");
    gpu_sim.Copy(keys[0], K_all, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_K_all = matrix_memory_allocator.Allocate("new_K_all");
      gpu_sim.Concat(K_all, keys[j], new_K_all, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_all);
      K_all = new_K_all;
    }
    
    // Create V_all by concatenating values
    Matrix* V_all = matrix_memory_allocator.Allocate("V_all");
    gpu_sim.Copy(values[0], V_all, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* new_V_all = matrix_memory_allocator.Allocate("new_V_all");
      gpu_sim.Concat(V_all, values[j], new_V_all, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V_all);
      V_all = new_V_all;
    }
    
    // Transpose K_all
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Compute scores = Q @ K_all^T
    Matrix* scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, K_all, scores);
    gpu_sim.ReleaseMatrix(K_all);
    
    // Compute Softmax(scores) row-wise
    Matrix* exp_scores = matrix_memory_allocator.Allocate("exp_scores");
    gpu_sim.MatExp(scores, exp_scores);
    gpu_sim.ReleaseMatrix(scores);
    
    // Compute row sums and normalize
    Matrix* weights = nullptr;
    for (size_t j = 0; j < m; ++j) {
      Matrix* row_j = matrix_memory_allocator.Allocate("row_j");
      gpu_sim.GetRow(exp_scores, j, row_j, kInSharedMemory);
      
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_j, row_sum);
      
      Matrix* normalized_row_j = matrix_memory_allocator.Allocate("normalized_row_j");
      gpu_sim.MatDiv(row_j, row_sum, normalized_row_j);
      
      gpu_sim.ReleaseMatrix(row_j);
      gpu_sim.ReleaseMatrix(row_sum);
      
      if (weights == nullptr) {
        weights = normalized_row_j;
      } else {
        Matrix* new_weights = matrix_memory_allocator.Allocate("new_weights");
        gpu_sim.Concat(weights, normalized_row_j, new_weights, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(weights);
        gpu_sim.ReleaseMatrix(normalized_row_j);
        weights = new_weights;
      }
    }
    gpu_sim.ReleaseMatrix(exp_scores);
    
    // Compute output = weights @ V_all
    Matrix* output = matrix_memory_allocator.Allocate("output");
    gpu_sim.MatMul(weights, V_all, output);
    gpu_sim.ReleaseMatrix(weights);
    gpu_sim.ReleaseMatrix(V_all);
    
    // Move output to HBM
    gpu_sim.MoveMatrixToGpuHbm(output);
    
    // Run the simulator once
    gpu_sim.Run(false, &matrix_memory_allocator);
    
    // Commit answer
    rater.CommitAnswer(*output);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
