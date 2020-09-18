# Multi-thread Multi-Head GPU Coordinates Mapping with Shared SLAB Router and Memory

## What's news?
- Light Head: the table head has been shrinked 32 times.
- Multi-Head: one slab router and memory, multi table head.
- Singleton: reuse slab router and memory by making SlabAlloc singleton,
  preserve allocating and releasing large GPU memory frequently.
- Random Ring Hash: one random number per table head as the memory block
  offset to make the worse space usage best and uniform.
- Compact slab memory layout design: support any dim of coordinate with
  high gpu memory usage(around 50% ~ 100%), without lossing speed.

- Bug free: No insertion while deletion bug(due to read and write
  sequence in lock-free logic) in origin SlabHash(Saman
  Ashkiani version).
            No insertion bug(do not support duplicate insertion due to lock-free logic or wrap programing logic) and Remove bug(due to wrap programing logic) in GPU CoordinateHash(Wei Dong version).

## Usage example:
more details in test_unique_with_remove_multithread_with_query_coords.cu

```
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
```

## TODO


1. General improvment: [Easy]

  - move GPU Memory configuration into template
  - change pass-by-value to pass-by-pointer for `Key`
  - support any `Value` type internelly(only support int currently).

2. Custom it to specific usage:

  - custom kernel
  - custom memory handling

## Best Practice

Features supported by embedded it into MinkowskiEngine

  - Mapping As Indices
  - Iterate As Insertion
  - Insertion As Search
  - **Accelerate any sparse, including query-ball in pointcloud and
    pv-rcnn.**

## Acknowledge

  - (cuda_unordered_map)[https://github.com/theNded/cuda_unordered_map]
  - (SlabHash)[https://github.com/owensgroup/SlabHash]
  * [Saman Ashkiani, Martin Farach-Colton, John Owens, *A Dynamic Hash Table for the GPU*, 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)](https://ieeexplore.ieee.org/abstract/document/8425196)
