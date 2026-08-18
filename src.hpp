#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  Matrix *key_columns = nullptr; // d by number of keys seen so far
  Matrix *value_rows = nullptr;  // number of values seen so far by d

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *query = rater.GetNextQuery();

    // Put the new cache entries before the query in the IO queue.  This lets
    // cache construction overlap most of the query transfer.
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    gpu_sim.MoveMatrixToSharedMem(query);
    gpu_sim.Transpose(keys[i], kInSharedMemory);

    if (i == 0) {
      key_columns = keys[i];
      value_rows = values[i];
    } else {
      Matrix *new_keys =
          matrix_memory_allocator.Allocate("key_columns_" + std::to_string(i));
      gpu_sim.Concat(key_columns, keys[i], new_keys, 1, kInSharedMemory);
      gpu_sim.ReleaseMatrix(key_columns);
      gpu_sim.ReleaseMatrix(keys[i]);
      key_columns = new_keys;

      Matrix *new_values =
          matrix_memory_allocator.Allocate("value_rows_" + std::to_string(i));
      gpu_sim.Concat(value_rows, values[i], new_values, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(value_rows);
      gpu_sim.ReleaseMatrix(values[i]);
      value_rows = new_values;
    }

    Matrix *scores =
        matrix_memory_allocator.Allocate("scores_" + std::to_string(i));
    Matrix *exponentials =
        matrix_memory_allocator.Allocate("exponentials_" + std::to_string(i));
    gpu_sim.MatMul(query, key_columns, scores);
    gpu_sim.MatExp(scores, exponentials);
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(query);

    // Sum/MatDiv are scalar-reduction operations, so normalize one row at a
    // time.  Multiplying each normalized row immediately also avoids keeping
    // both the complete probability matrix and the answer in SRAM.
    Matrix *answer = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      Matrix *weights = matrix_memory_allocator.Allocate(
          "exp_row_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *denominator = matrix_memory_allocator.Allocate(
          "denominator_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *normalized = matrix_memory_allocator.Allocate(
          "normalized_" + std::to_string(i) + "_" + std::to_string(row));
      Matrix *output_row = matrix_memory_allocator.Allocate(
          "output_row_" + std::to_string(i) + "_" + std::to_string(row));

      gpu_sim.GetRow(exponentials, row, weights, kInSharedMemory);
      gpu_sim.Sum(weights, denominator);
      gpu_sim.MatDiv(weights, denominator, normalized);
      gpu_sim.MatMul(normalized, value_rows, output_row);
      gpu_sim.ReleaseMatrix(weights);
      gpu_sim.ReleaseMatrix(denominator);
      gpu_sim.ReleaseMatrix(normalized);

      if (row == 0) {
        answer = output_row;
      } else {
        Matrix *joined = matrix_memory_allocator.Allocate(
            "answer_" + std::to_string(i) + "_" + std::to_string(row));
        gpu_sim.Concat(answer, output_row, joined, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        gpu_sim.ReleaseMatrix(output_row);
        answer = joined;
      }
    }
    gpu_sim.ReleaseMatrix(exponentials);
    gpu_sim.MoveMatrixToGpuHbm(answer);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
