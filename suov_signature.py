import torch
import hashlib
import random

print("==========================================================")
print("=== SUOV: Sedenionic Unbalanced Oil & Vinegar Signatures =")
print("==========================================================")

P_PRIME = 2147483647 # Mersenne Prime 2^31 - 1

# --- 1. GF(p) UTILITIES & SAFE LINEAR ALGEBRA ---
def mod_p(tensor):
    return torch.remainder(tensor, P_PRIME)

def matmul_mod_p(A, B):
    """Safe Matrix Multiplication using Python's arbitrary-precision ints to avoid PyTorch int64 overflow."""
    A_list = A.tolist()
    B_list = B.tolist()
    m = len(A_list)
    n = len(A_list[0])
    cols = len(B_list[0])
    C_list = [[0]*cols for _ in range(m)]
    for i in range(m):
        for j in range(cols):
            val = 0
            for k in range(n):
                val = (val + A_list[i][k] * B_list[k][j]) % P_PRIME
            C_list[i][j] = val
    return torch.tensor(C_list, dtype=torch.int64)

def invert_matrix_gf(M_tensor, p):
    n = M_tensor.size(0)
    M = M_tensor.tolist()
    for i in range(n): M[i] += [1 if i == j else 0 for j in range(n)]
    
    for i in range(n):
        pivot = i
        while pivot < n and M[pivot][i] == 0: pivot += 1
        if pivot == n: raise ValueError("Singular matrix generated. Try again.")
        M[i], M[pivot] = M[pivot], M[i]
        inv = pow(int(M[i][i]), -1, p)
        for j in range(i, 2*n): M[i][j] = (M[i][j] * inv) % p
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, 2*n): M[k][j] = (M[k][j] - factor * M[i][j]) % p
    return torch.tensor([row[n:] for row in M], dtype=torch.int64)

def solve_linear_system_gf(A_tensor, b_tensor, p):
    n = A_tensor.size(0)
    M = A_tensor.tolist()
    b = b_tensor.tolist()
    for i in range(n): M[i].append(b[i][0])
    
    for i in range(n):
        pivot = i
        while pivot < n and M[pivot][i] == 0: pivot += 1
        if pivot == n: return None 
        M[i], M[pivot] = M[pivot], M[i]
        inv = pow(int(M[i][i]), -1, p)
        for j in range(i, n + 1): M[i][j] = (M[i][j] * inv) % p
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1): M[k][j] = (M[k][j] - factor * M[i][j]) % p
    return torch.tensor([[M[i][n]] for i in range(n)], dtype=torch.int64)

# --- 2. GF(p) SEDENION ALGEBRA (OVERFLOW PROOF) ---
def octonion_mul_gf(a, b):
    a_l = a.tolist()
    b_l = b.tolist()
    c_l = [[0]*8 for _ in range(len(a_l))]
    for i in range(len(a_l)):
        a0,a1,a2,a3,a4,a5,a6,a7 = a_l[i]
        b0,b1,b2,b3,b4,b5,b6,b7 = b_l[i]
        # Modulo applied at every step to prevent tensor explosions
        c_l[i][0] = (a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7) % P_PRIME
        c_l[i][1] = (a0*b1 + a1*b0 + a2*b4 + a3*b7 - a4*b2 + a5*b6 - a6*b5 - a7*b3) % P_PRIME
        c_l[i][2] = (a0*b2 - a1*b4 + a2*b0 + a3*b5 + a4*b1 - a5*b3 + a6*b7 - a7*b6) % P_PRIME
        c_l[i][3] = (a0*b3 - a1*b7 - a2*b5 + a3*b0 + a4*b6 + a5*b2 - a6*b4 + a7*b1) % P_PRIME
        c_l[i][4] = (a0*b4 + a1*b2 - a2*b1 - a3*b6 + a4*b0 + a5*b7 + a6*b3 - a7*b5) % P_PRIME
        c_l[i][5] = (a0*b5 - a1*b6 + a2*b3 - a3*b2 - a4*b7 + a5*b0 + a6*b1 + a7*b4) % P_PRIME
        c_l[i][6] = (a0*b6 + a1*b5 - a2*b7 + a3*b4 - a4*b3 - a5*b1 + a6*b0 + a7*b2) % P_PRIME
        c_l[i][7] = (a0*b7 + a1*b3 + a2*b6 - a3*b1 + a4*b5 - a5*b4 - a6*b2 + a7*b0) % P_PRIME
    return torch.tensor(c_l, dtype=torch.int64)

def octonion_conj_gf(a):
    c = a.clone()
    c[:, 1:] = -c[:, 1:]
    return mod_p(c)

def sedenion_mul_gf(x, y):
    x_a, x_b = x[:, 0:8], x[:, 8:16]
    y_a, y_b = y[:, 0:8], y[:, 8:16]
    z_a = mod_p(octonion_mul_gf(x_a, y_a) - octonion_mul_gf(octonion_conj_gf(y_b), x_b))
    z_b = mod_p(octonion_mul_gf(y_b, x_a) + octonion_mul_gf(x_b, octonion_conj_gf(y_a)))
    return torch.cat([z_a, z_b], dim=1)

# --- 3. CRYPTOGRAPHIC PRIMITIVES ---
def hash_to_sedenion(message: str) -> torch.Tensor:
    h = hashlib.sha3_256(message.encode()).digest()
    vec = [int.from_bytes(h[i:i+2], 'big') for i in range(0, 32, 2)]
    return mod_p(torch.tensor([vec], dtype=torch.int64))

class SUOV_Keypair:
    def __init__(self):
        while True:
            try:
                self.T = torch.randint(0, P_PRIME, (16, 16), dtype=torch.int64)
                self.T_inv = invert_matrix_gf(self.T, P_PRIME)
                break
            except ValueError: pass

        self.M_sec = torch.randint(0, P_PRIME, (16, 16), dtype=torch.int64)
        self.M_sec[:, 8:16] = 0 
        
        # M_pub = T * M_sec * T^-1
        temp = matmul_mod_p(self.T, self.M_sec)
        self.M_pub = matmul_mod_p(temp, self.T_inv)

# --- 4. SIGNING & VERIFICATION ALGORITHMS ---
def sign_message(keypair: SUOV_Keypair, message: str) -> torch.Tensor:
    P_geom = hash_to_sedenion(message)
    
    while True:
        V = torch.zeros((1, 16), dtype=torch.int64)
        V[0, 0:8] = torch.randint(0, P_PRIME, (8,))
        
        # Translate Vinegar to Public Basis early
        TV = matmul_mod_p(keypair.T, V.T).T
        
        # Calculate K = T * M_sec * V
        M_sec_V = matmul_mod_p(keypair.M_sec, V.T).T
        K = matmul_mod_p(keypair.T, M_sec_V.T).T
        
        TV_P = sedenion_mul_gf(TV, P_geom)
        C = sedenion_mul_gf(TV_P, K)
        b_target = mod_p(-C[0, 0:8]).unsqueeze(1)
        
        A = torch.zeros((8, 8), dtype=torch.int64)
        for i in range(8):
            E_i = torch.zeros((1, 16), dtype=torch.int64)
            E_i[0, 8+i] = 1
            
            T_Ei = matmul_mod_p(keypair.T, E_i.T).T 
            T_Ei_P = sedenion_mul_gf(T_Ei, P_geom)
            Col_i = sedenion_mul_gf(T_Ei_P, K)
            A[:, i] = Col_i[0, 0:8]
            
        O_solved = solve_linear_system_gf(A, b_target, P_PRIME)
        if O_solved is None:
            continue
            
        S_prime = V.clone()
        S_prime[0, 8:16] = O_solved.squeeze()
        
        Signature = matmul_mod_p(keypair.T, S_prime.T).T
        return Signature

def verify_signature(M_pub: torch.Tensor, signature: torch.Tensor, message: str) -> bool:
    P_geom = hash_to_sedenion(message)
    
    SP = sedenion_mul_gf(signature, P_geom)
    MS = matmul_mod_p(M_pub, signature.T).T
    
    Result = sedenion_mul_gf(SP, MS)
    
    residual = torch.sum(torch.abs(Result[0, 0:8])).item()
    return residual == 0

# --- 5. EXECUTION & DEMO ---
print("[System] Generating SUOV Keypair (1 KB Public Key)...")
keys = SUOV_Keypair()

msg = "Synergeia Super-Exponential Consensus Payload"
print(f"[System] Original Message: '{msg}'")

print("[System] Alice is signing the message using her hidden Isotropic Subspace...")
sig = sign_message(keys, msg)
print(f"         Generated 16D Signature: {sig[0, :4].numpy()}... (truncated)")

print("\n[System] Verifying Authentic Signature...")
is_valid = verify_signature(keys.M_pub, sig, msg)
print(f"         Result: {'[VALID]' if is_valid else '[INVALID]'} (Trapdoor Annihilated)")

print("\n[System] Eve attempts Existential Forgery (Altering the message)...")
evil_msg = "Synergeia Super-Exponential Consensus Payload - Send funds to Eve"
is_valid_forgery = verify_signature(keys.M_pub, sig, evil_msg)
print(f"         Result: {'[VALID]' if is_valid_forgery else '[INVALID - FORGERY BLOCKED]'} (Trapdoor Failed)")

print("\n==========================================================")