# Sedenion Cryptographic Stress Test

This repository contains a GPU-accelerated simulation of the Sedenion distributed consensus protocol. It demonstrates the use of 16-dimensional Sedenionic zero-divisors as cryptographic trapdoors. 

The included demo generates **1,000,000 randomized 16D forged keys** and evaluates them against a broadcasted state simultaneously using batched CUDA tensor cores, effectively brute-forcing the non-associative geometry in milliseconds.

## Project Structure
* `sedenion_crypto.py` - The main Python executable for the 1-million payload stress test.
* `setup.py` - The PyTorch C++ Extension build script.
* `src/associator.cu` - The native CUDA kernel handling batched Cayley-Dickson multiplication.
* `src/bindings.cpp` - PyBind11 bindings to expose the CUDA kernels to Python.

## Prerequisites
* **OS:** Windows / Linux
* **Python:** 3.12+
* **CUDA Toolkit:** 12.1
* **PyTorch:** Must be installed with CUDA support.

## Build Instructions
Before running the simulation, you must compile the custom C++ CUDA extension (`sedenion_ops`) to bind the 16D mathematical kernels to Python.

1. Open your terminal in the root directory containing `setup.py`.
2. Run the build command:
   ```bash
   python setup.py install --user

```

*(Note for Windows users: You must have the Visual Studio C++ Build Tools installed for the compiler to successfully build the extension).*

## Running the Demo

Once the extension is compiled, simply execute the stress test script:

```bash
python sedenion_crypto.py

```

**Expected Output:** The script will evaluate all 1,000,000 keys simultaneously on the GPU. You will see the execution time (typically < 5ms) and the massive geometric residual of the rejected forged keys.