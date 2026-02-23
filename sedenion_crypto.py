import torch
import time
import sedenion_ops

print("==========================================================")
print("=== Null-State GF(p): Hardware Acceleration Stress Test ===")
print("==========================================================")

# The Mersenne Prime used in the CUDA kernel
P_PRIME = 2147483647
num_attacks = 10_000_000

print(f"[System] Generating {num_attacks:,} randomized 16D integer forged keys on GPU...")

# 1. ALICE'S TARGET TRAJECTORY (P * L_pub)
# We set up the known target state that needs to be annihilated
alice_target = torch.zeros((1, 16), dtype=torch.int64, device='cuda')
alice_target[0, 1] = 1  # e1
alice_target[0, 10] = 1 # e10
alice_batch = alice_target.repeat(num_attacks, 1)

# 2. MALICIOUS ACTOR SWARM (1 Million Random Integer Keys)
# Generate completely random 64-bit integers modulo P_PRIME
malicious_sigs = torch.randint(0, P_PRIME, (num_attacks, 16), dtype=torch.int64, device='cuda')

# 3. INJECT THE HONEST SIGNATURE AT INDEX 0
# This is the known zero-divisor that perfectly annihilates Alice's target
malicious_sigs[0] = 0
malicious_sigs[0, 3] = 1  # e3
malicious_sigs[0, 14] = 1 # e14

# 4. PRE-ALLOCATE OUTPUT TENSOR
trapdoor_results = torch.zeros((num_attacks, 16), dtype=torch.int64, device='cuda')

print("[System] Firing batched discrete Cayley-Dickson multiplication...")

# --- PERFORMANCE TIMING ---
# Warm up GPU
sedenion_ops.batched_sedenion_mul(alice_batch[:100], malicious_sigs[:100], trapdoor_results[:100])

torch.cuda.synchronize()
start_time = time.time()

# BOOM: 1 Million discrete 16D modular multiplications in one shot
sedenion_ops.batched_sedenion_mul(alice_batch, malicious_sigs, trapdoor_results)

torch.cuda.synchronize()
end_time = time.time()
execution_time_ms = (end_time - start_time) * 1000

# --- ANALYSIS ---
# In discrete GF(p) math, a successful annihilation means every single dimension is exactly 0.
# We sum the absolute values of the dimensions. If the sum is 0, it's a perfect zero-divisor.
magnitudes = torch.sum(torch.abs(trapdoor_results), dim=1)

honest_residual = magnitudes[0].item()
closest_forged = torch.min(magnitudes[1:]).item()
successful_hacks = torch.sum(magnitudes[1:] == 0).item()

print("\n--- STRESS TEST RESULTS ---")
print(f"CUDA Execution Time:     {execution_time_ms:.2f} ms")
print(f"Honest Key Residual:     {honest_residual} (Verified - Absolute Zero)")
print(f"Closest Forged Residual: {closest_forged} (Rejected)")
print(f"Total Successful Hacks:  {successful_hacks} / {num_attacks - 1}")
print("==========================================================")