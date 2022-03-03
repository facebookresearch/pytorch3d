/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file provides utilities for dispatching to specialized versions of
// functions. This is especially useful for CUDA kernels, since specializing
// them to particular input sizes can often allow the compiler to unroll loops
// and place arrays into registers, which can give huge performance speedups.
//
// As an example, suppose we have the following function which is specialized
// based on a compile-time int64_t value:
//
// template<typename T, int64_t x>
// struct SquareOffset {
//   static void run(T y) {
//     T val = x * x + y;
//     std::cout << val << std::endl;
//   }
// }
//
// This function takes one compile-time argument x, and one run-time argument y.
// We might want to compile specialized versions of this for x=0, x=1, etc and
// then dispatch to the correct one based on the runtime value of x.
// One simple way to achieve this is with a lookup table:
//
// template<typename T>
// void DispatchSquareOffset(const int64_t x, T y) {
//   if (x == 0) {
//     SquareOffset<T, 0>::run(y);
//   } else if (x == 1) {
//     SquareOffset<T, 1>::run(y);
//   } else if (x == 2) {
//     SquareOffset<T, 2>::run(y);
//   }
// }
//
// This function takes both x and y as run-time arguments, and dispatches to
// different specialized versions of SquareOffset based on the run-time value
// of x. This works, but it's tedious and error-prone. If we want to change the
// set of x values for which we provide compile-time specializations, then we
// will need to do a lot of tedius editing of the dispatch function. Also, if we
// want to provide compile-time specializations for another function other than
// SquareOffset, we will need to duplicate the entire lookup table.
//
// To solve these problems, we can use the DispatchKernel1D function provided by
// this file instead:
//
// template<typename T>
// void DispatchSquareOffset(const int64_t x, T y) {
//     constexpr int64_t xmin = 0;
//     constexpr int64_t xmax = 2;
//     DispatchKernel1D<SquareOffset, T, xmin, xmax>(x, y);
// }
//
// DispatchKernel1D uses template metaprogramming to compile specialized
// versions of SquareOffset for all values of x with xmin <= x <= xmax, and
// then dispatches to the correct one based on the run-time value of x. If we
// want to change the range of x values for which SquareOffset is specialized
// at compile-time, then all we have to do is change the values of the
// compile-time constants xmin and xmax.
//
// This file also allows us to similarly dispatch functions that depend on two
// compile-time int64_t values, using the DispatchKernel2D function like this:
//
// template<typename T, int64_t x, int64_t y>
// struct Sum {
//   static void run(T z, T w) {
//     T val = x + y + z + w;
//     std::cout << val << std::endl;
//   }
// }
//
// template<typename T>
// void DispatchSum(const int64_t x, const int64_t y, int z, int w) {
//   constexpr int64_t xmin = 1;
//   constexpr int64_t xmax = 3;
//   constexpr int64_t ymin = 2;
//   constexpr int64_t ymax = 5;
//   DispatchKernel2D<Sum, T, xmin, xmax, ymin, ymax>(x, y, z, w);
// }
//
// Like its 1D counterpart, DispatchKernel2D uses template metaprogramming to
// compile specialized versions of sum for all values of (x, y) with
// xmin <= x <= xmax and ymin <= y <= ymax, then dispatches to the correct
// specialized version based on the runtime values of x and y.

// Define some helper structs in an anonymous namespace.
namespace {

// 1D dispatch: general case.
// Kernel is the function we want to dispatch to; it should take a typename and
// an int64_t as template args, and it should define a static void function
// run which takes any number of arguments of any type.
// In order to dispatch, we will take an additional template argument curN,
// and increment it via template recursion until it is equal to the run-time
// argument N.
template <
    template <typename, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t curN,
    typename... Args>
struct DispatchKernelHelper1D {
  static void run(const int64_t N, Args... args) {
    if (curN == N) {
      // The compile-time value curN is equal to the run-time value N, so we
      // can dispatch to the run method of the Kernel.
      Kernel<T, curN>::run(args...);
    } else if (curN < N) {
      // Increment curN via template recursion
      DispatchKernelHelper1D<Kernel, T, minN, maxN, curN + 1, Args...>::run(
          N, args...);
    }
    // We shouldn't get here -- throw an error?
  }
};

// 1D dispatch: Specialization when curN == maxN
// We need this base case to avoid infinite template recursion.
template <
    template <typename, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    typename... Args>
struct DispatchKernelHelper1D<Kernel, T, minN, maxN, maxN, Args...> {
  static void run(const int64_t N, Args... args) {
    if (N == maxN) {
      Kernel<T, maxN>::run(args...);
    }
    // We shouldn't get here -- throw an error?
  }
};

// 2D dispatch, general case.
// This is similar to the 1D case: we take additional template args curN and
// curM, and increment them via template recursion until they are equal to
// the run-time values of N and M, at which point we dispatch to the run
// method of the kernel.
template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t curN,
    int64_t minM,
    int64_t maxM,
    int64_t curM,
    typename... Args>
struct DispatchKernelHelper2D {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (curN == N && curM == M) {
      Kernel<T, curN, curM>::run(args...);
    } else if (curN < N && curM < M) {
      // Increment both curN and curM. This isn't strictly necessary; we could
      // just increment one or the other at each step. But this helps to cut
      // on the number of recursive calls we make.
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN + 1,
          minM,
          maxM,
          curM + 1,
          Args...>::run(N, M, args...);
    } else if (curN < N) {
      // Increment curN only
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN + 1,
          minM,
          maxM,
          curM,
          Args...>::run(N, M, args...);
    } else if (curM < M) {
      // Increment curM only
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN,
          minM,
          maxM,
          curM + 1,
          Args...>::run(N, M, args...);
    }
  }
};

// 2D dispatch, specialization for curN == maxN
template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t minM,
    int64_t maxM,
    int64_t curM,
    typename... Args>
struct DispatchKernelHelper2D<
    Kernel,
    T,
    minN,
    maxN,
    maxN,
    minM,
    maxM,
    curM,
    Args...> {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (maxN == N && curM == M) {
      Kernel<T, maxN, curM>::run(args...);
    } else if (curM < maxM) {
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          maxN,
          minM,
          maxM,
          curM + 1,
          Args...>::run(N, M, args...);
    }
    // We should not get here -- throw an error?
  }
};

// 2D dispatch, specialization for curM == maxM
template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t curN,
    int64_t minM,
    int64_t maxM,
    typename... Args>
struct DispatchKernelHelper2D<
    Kernel,
    T,
    minN,
    maxN,
    curN,
    minM,
    maxM,
    maxM,
    Args...> {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (curN == N && maxM == M) {
      Kernel<T, curN, maxM>::run(args...);
    } else if (curN < maxN) {
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN + 1,
          minM,
          maxM,
          maxM,
          Args...>::run(N, M, args...);
    }
    // We should not get here -- throw an error?
  }
};

// 2D dispatch, specialization for curN == maxN, curM == maxM
template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t minM,
    int64_t maxM,
    typename... Args>
struct DispatchKernelHelper2D<
    Kernel,
    T,
    minN,
    maxN,
    maxN,
    minM,
    maxM,
    maxM,
    Args...> {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (maxN == N && maxM == M) {
      Kernel<T, maxN, maxM>::run(args...);
    }
    // We should not get here -- throw an error?
  }
};

} // namespace

// This is the function we expect users to call to dispatch to 1D functions
template <
    template <typename, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    typename... Args>
void DispatchKernel1D(const int64_t N, Args... args) {
  if (minN <= N && N <= maxN) {
    // Kick off the template recursion by calling the Helper with curN = minN
    DispatchKernelHelper1D<Kernel, T, minN, maxN, minN, Args...>::run(
        N, args...);
  }
  // Maybe throw an error if we tried to dispatch outside the allowed range?
}

// This is the function we expect users to call to dispatch to 2D functions
template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t minM,
    int64_t maxM,
    typename... Args>
void DispatchKernel2D(const int64_t N, const int64_t M, Args... args) {
  if (minN <= N && N <= maxN && minM <= M && M <= maxM) {
    // Kick off the template recursion by calling the Helper with curN = minN
    // and curM = minM
    DispatchKernelHelper2D<
        Kernel,
        T,
        minN,
        maxN,
        minN,
        minM,
        maxM,
        minM,
        Args...>::run(N, M, args...);
  }
  // Maybe throw an error if we tried to dispatch outside the specified range?
}
