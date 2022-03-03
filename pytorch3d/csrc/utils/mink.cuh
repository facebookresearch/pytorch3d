/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#define MINK_H

#include "index_utils.cuh"

// A data structure to keep track of the smallest K keys seen so far as well
// as their associated values, intended to be used in device code.
// This data structure doesn't allocate any memory; keys and values are stored
// in arrays passed to the constructor.
//
// The implementation is generic; it can be used for any key type that supports
// the < operator, and can be used with any value type.
//
// Example usage:
//
// float keys[K];
// int values[K];
// MinK<float, int> mink(keys, values, K);
// for (...) {
//   // Produce some key and value from somewhere
//   mink.add(key, value);
// }
// mink.sort();
//
// Now keys and values store the smallest K keys seen so far and the values
// associated to these keys:
//
// for (int k = 0; k < K; ++k) {
//   float key_k = keys[k];
//   int value_k = values[k];
// }
template <typename key_t, typename value_t>
class MinK {
 public:
  // Constructor.
  //
  // Arguments:
  //   keys: Array in which to store keys
  //   values: Array in which to store values
  //   K: How many values to keep track of
  __device__ MinK(key_t* keys, value_t* vals, int K)
      : keys(keys), vals(vals), K(K), _size(0) {}

  // Try to add a new key and associated value to the data structure. If the key
  // is one of the smallest K seen so far then it will be kept; otherwise it
  // it will not be kept.
  //
  // This takes O(1) operations if the new key is not kept, or if the structure
  // currently contains fewer than K elements. Otherwise this takes O(K) time.
  //
  // Arguments:
  //   key: The key to add
  //   val: The value associated to the key
  __device__ __forceinline__ void add(const key_t& key, const value_t& val) {
    if (_size < K) {
      keys[_size] = key;
      vals[_size] = val;
      if (_size == 0 || key > max_key) {
        max_key = key;
        max_idx = _size;
      }
      _size++;
    } else if (key < max_key) {
      keys[max_idx] = key;
      vals[max_idx] = val;
      max_key = key;
      for (int k = 0; k < K; ++k) {
        key_t cur_key = keys[k];
        if (cur_key > max_key) {
          max_key = cur_key;
          max_idx = k;
        }
      }
    }
  }

  // Get the number of items currently stored in the structure.
  // This takes O(1) time.
  __device__ __forceinline__ int size() {
    return _size;
  }

  // Sort the items stored in the structure using bubble sort.
  // This takes O(K^2) time.
  __device__ __forceinline__ void sort() {
    for (int i = 0; i < _size - 1; ++i) {
      for (int j = 0; j < _size - i - 1; ++j) {
        if (keys[j + 1] < keys[j]) {
          key_t key = keys[j];
          value_t val = vals[j];
          keys[j] = keys[j + 1];
          vals[j] = vals[j + 1];
          keys[j + 1] = key;
          vals[j + 1] = val;
        }
      }
    }
  }

 private:
  key_t* keys;
  value_t* vals;
  int K;
  int _size;
  key_t max_key;
  int max_idx;
};

// This is a version of MinK that only touches the arrays using static indexing
// via RegisterIndexUtils. If the keys and values are stored in thread-local
// arrays, then this may allow the compiler to place them in registers for
// fast access.
//
// This has the same API as RegisterMinK, but doesn't support sorting.
// We found that sorting via RegisterIndexUtils gave very poor performance,
// and suspect it may have prevented the compiler from placing the arrays
// into registers.
template <typename key_t, typename value_t, int K>
class RegisterMinK {
 public:
  __device__ RegisterMinK(key_t* keys, value_t* vals)
      : keys(keys), vals(vals), _size(0) {}

  __device__ __forceinline__ void add(const key_t& key, const value_t& val) {
    if (_size < K) {
      RegisterIndexUtils<key_t, K>::set(keys, _size, key);
      RegisterIndexUtils<value_t, K>::set(vals, _size, val);
      if (_size == 0 || key > max_key) {
        max_key = key;
        max_idx = _size;
      }
      _size++;
    } else if (key < max_key) {
      RegisterIndexUtils<key_t, K>::set(keys, max_idx, key);
      RegisterIndexUtils<value_t, K>::set(vals, max_idx, val);
      max_key = key;
      for (int k = 0; k < K; ++k) {
        key_t cur_key = RegisterIndexUtils<key_t, K>::get(keys, k);
        if (cur_key > max_key) {
          max_key = cur_key;
          max_idx = k;
        }
      }
    }
  }

  __device__ __forceinline__ int size() {
    return _size;
  }

 private:
  key_t* keys;
  value_t* vals;
  int _size;
  key_t max_key;
  int max_idx;
};
