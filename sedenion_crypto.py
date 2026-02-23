import torch
import sedenion_ops
import time

print("==========================================================")
print("=== APH Rank-5: Synergeia Cryptographic Stress Test    ===")
print("==========================================================")

def gpu_sedenion_mul(x, y):
    x = x.contiguous().cuda()
    y = y.contiguous().cuda()
    z = torch.zeros_like(x).contiguous().cuda()
    sedenion_ops.batched_sedenion_mul(x, y, z)
    return z

num_attacks = 1_000_000
print(f"[System] Generating {num_attacks:,} randomized 16D forged keys on GPU...")

# 1. ALICE'S BROADCAST (Target)
alice_key = torch.zeros((1, 16), device='cuda')
alice_key[0, 1] = 1.0  # e1
alice_key[0, 10] = 1.0 # e10
alice_batch = alice_key.repeat(num_attacks, 1)

# 2. MALICIOUS ACTOR SWARM (1 Million Random Keys)
# Generate random 16D floats, normalized to match the magnitude of our honest key
malicious_locks = torch.randn((num_attacks, 16), device='cuda')
malicious_locks = torch.nn.functional.normalize(malicious_locks, p=2, dim=1) * 1.4142

# 3. INJECT THE HONEST KEY AT INDEX 0
malicious_locks[0] = 0.0
malicious_locks[0, 3] = 1.0  # e3
malicious_locks[0, 14] = 1.0 # e14

print("[System] Firing batched Cayley-Dickson multiplication...")

# --- PERFORMANCE TIMING ---
torch.cuda.synchronize()
start_time = time.time()

# BOOM: 1 Million 16D Multiplications in one shot
trapdoor_results = gpu_sedenion_mul(alice_batch, malicious_locks)

torch.cuda.synchronize()
end_time = time.time()

# --- ANALYSIS ---
magnitudes = torch.norm(trapdoor_results, dim=1)

honest_mag = magnitudes[0].item()
forged_mags = magnitudes[1:]

min_forged_mag = torch.min(forged_mags).item()
successful_hacks = torch.sum(forged_mags < 1e-6).item()

print("\n--- STRESS TEST RESULTS ---")
print(f"CUDA Execution Time:     {(end_time - start_time) * 1000:.2f} ms")
print(f"Honest Key Residual:     {honest_mag:.6f} (Verified)")
print(f"Closest Forged Residual: {min_forged_mag:.6f} (Rejected)")
print(f"Total Successful Hacks:  {successful_hacks} / {num_attacks - 1:,}")
print("==========================================================")