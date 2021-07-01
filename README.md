# CudaSamples
Cuda Samples to demonstrate understanding of CUDA and GPU Architecture.

- **CudaSamples** : Default project created with Visual Studio 2019. Its an addition program.
- **WarpSample** : This project demonstrates Warp creations. It prints warp allocation layout in different blocks.
- **ParallelReduction** : Different parallel reduction techniques are implemented in this sample. (Timing is mentioned in bracket)
  - Neighboured Pair Approach (78.89ms)
  - To remove divergence in above approach:
    - Force Add neighbours (58.8ms)
    - Interleaved Pair (61.1ms)
  - Combine blocks to reduce instruction load. Add blocks manually and then do interleaved pair on single block. (32.94ms for 2 block merging and 18.15ms for 4 block merging)
  - Wrap Unrolling : perform reduction on wraps to end up with only one wrap which has all elements added on respective thread. Then unroll that wrap manually to get rid of divergence in wrap reduction. (9.77ms)
  - Complete Unroll : We still have loop to reduce all wraps in a block into single wrap. If we know block dimension is power of 2, we can easily unroll this for loop manually to get rid of this. (9.25ms)
- **PinnedMemory** : Test run to allocate pinned memory. (TODO: Add profiled info)
- **ZeroCopyMemory** : Test run to allocate zero copy memory. (TODO: Add profile info)
- **UnifiedMemory**: Test run to allocate UnifiedMemory. (TODO: Add Profile Info)

## WorkInProgress
