#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t m = i + 1;  // number of key-value pairs in this round
    
    // Step 1: Concatenate keys[0..i] into K_all (shape [m, 512])
    gpu_sim.MoveMatrixToSharedMem(keys[0]);
    Matrix* K_all = matrix_memory_allocator.Allocate("K_all");
    gpu_sim.Copy(keys[0], K_all, kInSharedMemory);
    
    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix* new_K_all = matrix_memory_allocator.Allocate("new_K_all");
      gpu_sim.Concat(K_all, keys[j], new_K_all, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_all);
      K_all = new_K_all;
    }
    
    // Step 2: Concatenate values[0..i] into V_all (shape [m, 512])
    gpu_sim.MoveMatrixToSharedMem(values[0]);
    Matrix* V_all = matrix_memory_allocator.Allocate("V_all");
    gpu_sim.Copy(values[0], V_all, kInSharedMemory);
    
    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      Matrix* new_V_all = matrix_memory_allocator.Allocate("new_V_all");
      gpu_sim.Concat(V_all, values[j], new_V_all, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V_all);
      V_all = new_V_all;
    }
    
    // Step 3: Move query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Step 4: Transpose K_all (shape [m, 512] -> [512, m])
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Step 5: Compute scores = Q @ K_all^T (shape [m, 512] @ [512, m] = [m, m])
    Matrix* scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, K_all, scores);
    gpu_sim.ReleaseMatrix(K_all);
    
    // Step 6: Compute Softmax(scores) row-wise
    // Step 6a: exp(scores)
    Matrix* exp_scores = matrix_memory_allocator.Allocate("exp_scores");
    gpu_sim.MatExp(scores, exp_scores);
    gpu_sim.ReleaseMatrix(scores);
    
    // Step 6b: For each row, normalize by row sum
    Matrix* weights = nullptr;
    for (size_t j = 0; j < m; ++j) {
      // Get row j from exp_scores
      Matrix* row_j = matrix_memory_allocator.Allocate("row_j");
      gpu_sim.GetRow(exp_scores, j, row_j, kInSharedMemory);
      
      // Compute sum of row j
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_j, row_sum);
      
      // Divide row by its sum
      Matrix* normalized_row_j = matrix_memory_allocator.Allocate("normalized_row_j");
      gpu_sim.MatDiv(row_j, row_sum, normalized_row_j);
      
      gpu_sim.ReleaseMatrix(row_j);
      gpu_sim.ReleaseMatrix(row_sum);
      
      // Concatenate to form weights matrix
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
    
    // Step 7: Compute output = weights @ V_all (shape [m, m] @ [m, 512] = [m, 512])
    Matrix* output = matrix_memory_allocator.Allocate("output");
    gpu_sim.MatMul(weights, V_all, output);
    gpu_sim.ReleaseMatrix(weights);
    gpu_sim.ReleaseMatrix(V_all);
    
    // Step 8: Move output to HBM
    gpu_sim.MoveMatrixToGpuHbm(output);
    
    // Step 9: Run the simulator to execute all queued instructions
    gpu_sim.Run(false, &matrix_memory_allocator);
    
    // Step 10: Commit answer
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
