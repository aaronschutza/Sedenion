import torch

print("==========================================================")
print("=== Null-State: SDSA Payload Binding Prototype         ===")
print("==========================================================")

# --- 1. VECTORIZED CAYLEY-DICKSON ALGEBRA ---
def octonion_mul(a, b):
    c = torch.zeros_like(a)
    c[:, 0] = a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3] - a[:,4]*b[:,4] - a[:,5]*b[:,5] - a[:,6]*b[:,6] - a[:,7]*b[:,7]
    c[:, 1] = a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,4] + a[:,3]*b[:,7] - a[:,4]*b[:,2] + a[:,5]*b[:,6] - a[:,6]*b[:,5] - a[:,7]*b[:,3]
    c[:, 2] = a[:,0]*b[:,2] - a[:,1]*b[:,4] + a[:,2]*b[:,0] + a[:,3]*b[:,5] + a[:,4]*b[:,1] - a[:,5]*b[:,3] + a[:,6]*b[:,7] - a[:,7]*b[:,6]
    c[:, 3] = a[:,0]*b[:,3] - a[:,1]*b[:,7] - a[:,2]*b[:,5] + a[:,3]*b[:,0] + a[:,4]*b[:,6] + a[:,5]*b[:,2] - a[:,6]*b[:,4] + a[:,7]*b[:,1]
    c[:, 4] = a[:,0]*b[:,4] + a[:,1]*b[:,2] - a[:,2]*b[:,1] - a[:,3]*b[:,6] + a[:,4]*b[:,0] + a[:,5]*b[:,7] + a[:,6]*b[:,3] - a[:,7]*b[:,5]
    c[:, 5] = a[:,0]*b[:,5] - a[:,1]*b[:,6] + a[:,2]*b[:,3] - a[:,3]*b[:,2] - a[:,4]*b[:,7] + a[:,5]*b[:,0] + a[:,6]*b[:,1] + a[:,7]*b[:,4]
    c[:, 6] = a[:,0]*b[:,6] + a[:,1]*b[:,5] - a[:,2]*b[:,7] + a[:,3]*b[:,4] - a[:,4]*b[:,3] - a[:,5]*b[:,1] + a[:,6]*b[:,0] + a[:,7]*b[:,2]
    c[:, 7] = a[:,0]*b[:,7] + a[:,1]*b[:,3] + a[:,2]*b[:,6] - a[:,3]*b[:,1] + a[:,4]*b[:,5] - a[:,5]*b[:,4] - a[:,6]*b[:,2] + a[:,7]*b[:,0]
    return c

def octonion_conj(a):
    c = a.clone()
    c[:, 1:] = -c[:, 1:]
    return c

def sedenion_mul(x, y):
    x_a, x_b = x[:, 0:8], x[:, 8:16]
    y_a, y_b = y[:, 0:8], y[:, 8:16]
    z_a = octonion_mul(x_a, y_a) - octonion_mul(octonion_conj(y_b), x_b)
    z_b = octonion_mul(y_b, x_a) + octonion_mul(x_b, octonion_conj(y_a))
    return torch.cat([z_a, z_b], dim=1)

# --- 2. SETUP ALICE'S ACCOUNT ---
# For demonstration, we select a Public Lock and Payload that evaluate to our known zero-divisor
L_pub = torch.zeros((1, 16)); L_pub[0, 1] = 1.0; L_pub[0, 10] = 1.0

print("[Network] Alice's Public Lock (L_pub) is registered.")

# --- 3. HONEST TRANSACTION ---
print("\n--- SCENARIO 1: Honest Payload Binding ---")
# Alice creates a valid payload (e.g., "Send Bob 5 Tokens")
# For simplicity, we use the real identity e0 so P * L_pub = L_pub
P_honest = torch.zeros((1, 16)); P_honest[0, 0] = 1.0

# Alice mathematically derives her one-time signature S such that S * (P * L) = 0
Signature_S = torch.zeros((1, 16)); Signature_S[0, 3] = 1.0; Signature_S[0, 14] = 1.0
print(f"-> Alice broadcasts Tuple: [Signature_S, P_honest]")

# Network Verification: V = S * (P * L_pub)
X_target = sedenion_mul(P_honest, L_pub)
V_honest = sedenion_mul(Signature_S, X_target)
mag_honest = torch.norm(V_honest).item()

if mag_honest < 1e-6:
    print(f"-> [VERIFIED] Trapdoor hit Absolute Zero (Magnitude: {mag_honest:.6f}).")
    print("-> Transaction mathematically authorized.")

# --- 4. REPLAY ATTACK (EVE) ---
print("\n--- SCENARIO 2: The Replay Attack ---")
print("-> Eve intercepts Alice's Signature_S from the mempool.")
# Eve creates a forged payload (e.g., "Send Eve 500 Tokens")
P_evil = torch.zeros((1, 16)); P_evil[0, 5] = 1.0

print(f"-> Eve broadcasts Forged Tuple: [Signature_S, P_evil]")

# Network Verification: V = S * (P_evil * L_pub)
X_evil_target = sedenion_mul(P_evil, L_pub)
V_evil = sedenion_mul(Signature_S, X_evil_target)
mag_evil = torch.norm(V_evil).item()

if mag_evil > 1e-6:
    print(f"-> [REJECTED] Signature failed to annihilate forged trajectory!")
    print(f"-> Residual Energy Detected (Magnitude: {mag_evil:.4f}).")
    print("-> Forgery exposed. Payload dropped.")

print("\n[Telemetry] Target Trajectories:")
print(f"Honest (P * L_pub): {X_target[0].numpy().round(2)}")
print(f"Forged (P_evil * L_pub): {X_evil_target[0].numpy().round(2)}")
print("==========================================================")