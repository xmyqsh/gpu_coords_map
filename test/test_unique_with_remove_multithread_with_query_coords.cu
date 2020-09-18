#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <unordered_map>
#include <thread>
#include "cuda_unordered_map.h"
#include "coordinate.h"


void TEST_COORDS(int key_size, int seed) {
    const int D = 3;
//    const int D = 6;
    std::default_random_engine generator;
//    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::uniform_int_distribution<int> dist(0, 1000);

    // generate raw data (a bunch of data mimicking at::Tensor)
    std::cout << "generating data...\n";
    std::vector<int> input_coords(key_size * D);
    for (int i = 0; i < key_size * D; ++i) {
      input_coords[i] = dist(generator);
    }
    std::cout << "data generated\n";

    // convert raw data (at::Tensor) to std::vector
    // and prepare indices
    std::cout << "converting format...\n";
    std::vector<Coordinate<int, D>> insert_keys(key_size + key_size);
    std::memcpy(insert_keys.data(), input_coords.data(),
                sizeof(int) * key_size * D);

    // also make sure memcpy works correctly
    for (int i = 0; i < key_size; ++i) {
      /*
      for (int d = 0; d < D; ++d) {
        assert(input_coords[i * D + d] == insert_keys[i][d]);
      }
      */
      insert_keys[key_size + i] = insert_keys[key_size - 1 - i];
      insert_keys[key_size + i][i % D] += 1;
    }

    std::vector<int> index(key_size + key_size);
    std::iota(index.begin(), index.end(), 0);
//    std::random_device rd;
//    std::mt19937 rng(rd());
    std::mt19937 rng(seed);
    std::shuffle(index.begin(), index.end(), rng);
    for (int i = 0; i != index.size(); ++i) {
        insert_keys[i] = insert_keys[index[i]];
//        insert_keys[i].x[0] = i / 32 / (seed + 2);
//        for (int d = 1; d != 6; ++d) insert_keys[i].x[d] = -1;
    }

    std::vector<Coordinate<int, D>> super_insert_keys;
    super_insert_keys.resize(insert_keys.size() * 4);
//    super_insert_keys.resize(insert_keys.size() * 2);
    std::cout << "OK ?" << std::endl;
    for (int i = 0; i != 4; ++i) {
//    for (int i = 0; i != 2; ++i) {
        for (int j = 0; j != insert_keys.size(); ++j) {
            super_insert_keys[i * insert_keys.size() + j] = insert_keys[j];
        }
    }
    std::cout << "OK ??" << std::endl;

    std::cout << "conversion finished\n";

    // cpu groundtruth
    std::cout << "generating std::unordered_map ground truth hashtable...\n";
    std::unordered_map<Coordinate<int, D>, int, CoordinateHashFunc<int, D>> unordered_map;
    for (int i = 0; i < key_size * 2; ++i) {
      unordered_map[insert_keys[i]] = 1;
    }
    std::cout << "ground truth generated\n";

    std::vector<int> count(D);
    for (int d = 0; d != D; ++d) {
//      std::cout << "d: " << d << std::endl;
      for (const auto& item : unordered_map) {
          auto key = item.first;
          key[d] += 1;
          if (unordered_map.count(key)) {
            ++count[d];
            /*
              std::cout << key[0] << '\t'
                        << key[1] << '\t'
                        << key[2] << std::endl;
                        */
          }
      }
    }

    thrust::device_vector<int> cuda_count = count;

    // gpu test
    std::cout << "inserting to cuda::unordered_map...\n";

    thrust::device_vector<Coordinate<int, D>> cuda_insert_keys = super_insert_keys;

    cuda::unordered_map<Coordinate<int, D>, int> cuda_unordered_map(key_size * 8);
//    cuda::unordered_map<Coordinate<int, D>, int> cuda_unordered_map(key_size * 4);
    auto cnt_value = cuda_unordered_map.BulkBuild(cuda_insert_keys);
    auto cuda_query_results = cuda_unordered_map.Search(cuda_insert_keys);
    cuda_unordered_map.CountElems(cuda_count);
//    cuda_unordered_map.Remove(cuda_insert_keys);
    std::cout << "insertion finished\n";

    // query
    /*
    std::cout << "generating query_data...\n";
    std::vector<Coordinate<int, D>> cuda_query_keys(insert_keys.size());
    for (int i = 0; i < key_size; ++i) {
      if (i % 3 != 2) { // 2/3 is valid
        cuda_query_keys[i] = cuda_insert_keys[i];
      } else { // 1/3 is invalid
        cuda_query_keys[i] = Coordinate<int, D>::random(generator, dist);
      }
    }
    */
    /*
    std::cout << "query data generated\n";

    std::cout << "query from cuda::unordered_map...\n";
    auto cuda_query_results = cuda_unordered_map.Search(cuda_query_keys);
    cuda_unordered_map.Remove(cuda_insert_keys);
    std::cout << "query results generated\n";

    std::cout << "comparing query results against ground truth...\n";
    for (int i = 0; i < cuda_query_keys.size(); ++i) {
      auto iter = unordered_map.find(cuda_query_keys[i]);
      if (iter == unordered_map.end()) {
        assert(cuda_query_results.second[i] == 0);
      } else {
        assert(cuda_query_results.first[i] == iter->second);
      }
    }
    */

    /*
    for (int i = 0; i < insert_keys.size(); ++i) {
        //if (cuda_query_results.second[i] != 0) {
        if (cuda_query_results.second[i] == 0) {
            std::cout << "<--- " << i << std::endl;
            for (int d = 0; d != 6; ++d) std::cout << insert_keys[i][d] << '\t';
            std::cout << std::endl;
        }
    }
    */
    auto sIze_after = cuda_unordered_map.Size();
    std::cout << "cnt_value: " << cnt_value
    << "\tunordered_map.size(): " << unordered_map.size() << '\t' << key_size
//    << "\tSize: " << sIze
//    << "\tSize After Remove: " << sIze_after << std::endl;
    << std::endl;
    std::cout << "insertion finished\n";

//    assert(sIze_after == 0);

    std::cout << "query from cuda::unordered_map...\n";
//    auto cuda_query_results = cuda_unordered_map.Search(cuda_query_keys);
//    cuda_unordered_map.Remove(cuda_insert_keys);
    std::cout << "query results generated\n";

/*    auto */ cuda_query_results = cuda_unordered_map.Search(cuda_insert_keys);
/*
    for (int i = 0; i < insert_keys.size(); ++i) {
        if (cuda_query_results.second[i] != 0) {
        //if (cuda_query_results.second[i] == 0) {
            std::cout << "---> " << i << std::endl;
            for (int d = 0; d != 6; ++d) std::cout << insert_keys[i][d] << '\t';
            std::cout << std::endl;
        }
    }
*/
    std::cout << "TEST_COORDS() passed\n";
}

int main() {
  // stress test
  for (int j = 0; ; ++j) {
    std::cout << "@@@@@@@@@@@@@@ j: " << j << std::endl;

    std::vector<std::thread> vt;
    vt.reserve(4);
    for (int i = 0; i != 4; ++i) {
        vt.emplace_back(std::thread([i] { TEST_COORDS(2400000*2, i); std::cout << "Finish " << i << "th TEST_COORDS" << std::endl; }));
    }

    for (int i = 0; i != 4; ++i) {
        vt[i].join();
    }

    sleep(1);

  }
}
