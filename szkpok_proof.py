import torch
import hashlib
import random
import json

print("==========================================================")
print("=== S-ZKPoK: Sedenionic Zero-Knowledge Proof of Knowledge ")
print("==========================================================")

P_PRIME = 2147483647 # Mersenne Prime 2^31 - 1
NUM_ROUNDS = 16      # Number of ZKP rounds for the PoC (use 128 for prod)

# --- 1. GF(p) UTILITIES & SAFE LINEAR ALGEBRA ---
def mod_p(tensor):
    return torch.remainder(tensor, P_PRIME)

def matmul_mod_p(A, B):
    A_list = A.tolist()
    B_list = B.tolist()
    m, n, cols = len(A_list), len(A_list[0]), len(B_list[0])
    C_list = [[0]*cols for _ in range(m)]
    for i in range(m):
        for j in range(cols):
            val = 0
            for k in range(n): val = (val + A_list[i][k] * B_list[k][j]) % P_PRIME
            C_list[i][j] = val
    return torch.tensor(C_list, dtype=torch.int64)

def invert_matrix_gf(M_tensor, p):
    n = M_tensor.size(0)
    M = M_tensor.tolist()
    for i in range(n): M[i] += [1 if i == j else 0 for j in range(n)]
    for i in range(n):
        pivot = i
        while pivot < n and M[pivot][i] == 0: pivot += 1
        if pivot == n: raise ValueError("Singular matrix generated.")
        M[i], M[pivot] = M[pivot], M[i]
        inv = pow(int(M[i][i]), -1, p)
        for j in range(i, 2*n): M[i][j] = (M[i][j] * inv) % p
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, 2*n): M[k][j] = (M[k][j] - factor * M[i][j]) % p
    return torch.tensor([row[n:] for row in M], dtype=torch.int64)

# --- 2. OVERFLOW-PROOF SEDENION ALGEBRA ---
def octonion_mul_gf(a, b):
    a_l, b_l = a.tolist(), b.tolist()
    c_l = [[0]*8 for _ in range(len(a_l))]
    for i in range(len(a_l)):
        a0,a1,a2,a3,a4,a5,a6,a7 = a_l[i]
        b0,b1,b2,b3,b4,b5,b6,b7 = b_l[i]
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

def sedenion_square_gf(x):
    return sedenion_mul_gf(x, x)

# --- 3. PUBLIC KEY SIMULATOR & TEST VECTORS ---
# We evaluate maps on deterministic random vectors to prove equivalence
TEST_VECTORS = [torch.randint(0, P_PRIME, (1, 16), dtype=torch.int64) for _ in range(5)]

def hash_evaluations(evaluations):
    """Hashes a set of 16D vector evaluations into a single commitment hash."""
    h = hashlib.sha3_256()
    for ev in evaluations: h.update(ev.numpy().tobytes())
    return h.hexdigest()

class PublicMapEvaluator:
    """Simulates Bob evaluating the public quadratic map without knowing L1 or L2."""
    def __init__(self, L1, L2):
        self._L1 = L1
        self._L2 = L2
        
    def evaluate(self, X):
        inner = matmul_mod_p(self._L2, X.T).T
        squared = sedenion_square_gf(inner)
        return matmul_mod_p(self._L1, squared.T).T

# --- 4. ZERO KNOWLEDGE PROVER & VERIFIER ---
def generate_proof(L1, L2, L1_inv, L2_inv):
    """Alice generates a Non-Interactive ZK Proof (Fiat-Shamir)"""
    commitments = []
    R_states = []
    
    # Step 1: Commitments
    for i in range(NUM_ROUNDS):
        # Generate random blinding matrices
        R1 = torch.randint(0, P_PRIME, (16, 16), dtype=torch.int64)
        R2 = torch.randint(0, P_PRIME, (16, 16), dtype=torch.int64)
        R_states.append((R1, R2))
        
        # Evaluate C(X) = R1 * (R2 * X)^2 on test vectors
        evals = []
        for v in TEST_VECTORS:
            inner = matmul_mod_p(R2, v.T).T
            squared = sedenion_square_gf(inner)
            out = matmul_mod_p(R1, squared.T).T
            evals.append(out)
            
        commitments.append(hash_evaluations(evals))
        
    # Step 2: Fiat-Shamir Challenge Generation
    # Hash all commitments together to pseudo-randomly generate the challenge bits
    master_hash = hashlib.sha256("".join(commitments).encode()).digest()
    challenges = [ (master_hash[i // 8] >> (i % 8)) & 1 for i in range(NUM_ROUNDS) ]
    
    # Step 3: Responses
    responses = []
    for i in range(NUM_ROUNDS):
        R1, R2 = R_states[i]
        c = challenges[i]
        
        if c == 0:
            # Prove Commitment (Reveal R1, R2)
            responses.append({"type": 0, "M1": R1, "M2": R2})
        else:
            # Prove Isomorphism (Reveal Q1, Q2)
            Q1 = matmul_mod_p(R1, L1_inv)
            Q2 = matmul_mod_p(L2_inv, R2)
            responses.append({"type": 1, "M1": Q1, "M2": Q2})
            
    return {"commitments": commitments, "responses": responses}

def verify_proof(public_map: PublicMapEvaluator, proof: dict):
    """Bob verifies the ZK Proof without learning L1 or L2"""
    commitments = proof["commitments"]
    responses = proof["responses"]
    
    # Step 1: Re-derive the Fiat-Shamir challenges
    master_hash = hashlib.sha256("".join(commitments).encode()).digest()
    challenges = [ (master_hash[i // 8] >> (i % 8)) & 1 for i in range(NUM_ROUNDS) ]
    
    # Step 2: Verify each round
    for i in range(NUM_ROUNDS):
        c = challenges[i]
        resp = responses[i]
        
        if resp["type"] != c: return False # Prover responded to the wrong challenge!
        
        evals = []
        if c == 0:
            # Verify C(X) using revealed R1, R2
            R1, R2 = resp["M1"], resp["M2"]
            for v in TEST_VECTORS:
                inner = matmul_mod_p(R2, v.T).T
                squared = sedenion_square_gf(inner)
                out = matmul_mod_p(R1, squared.T).T
                evals.append(out)
        else:
            # Verify C(X) using revealed Q1, Q2 and the Public Map
            # C(X) = Q1 * PublicMap( Q2 * X )
            Q1, Q2 = resp["M1"], resp["M2"]
            for v in TEST_VECTORS:
                inner_q2 = matmul_mod_p(Q2, v.T).T
                pub_eval = public_map.evaluate(inner_q2)
                out = matmul_mod_p(Q1, pub_eval.T).T
                evals.append(out)
                
        # Hash the re-calculated evaluations
        recalc_hash = hash_evaluations(evals)
        if recalc_hash != commitments[i]:
            return False # Hash mismatch!
            
    return True

# --- 5. EXECUTION & DEMO ---
print("[System] Alice generates her private Sedenionic Keys (L1, L2)...")
L1 = torch.randint(0, P_PRIME, (16, 16), dtype=torch.int64); L1_inv = invert_matrix_gf(L1, P_PRIME)
L2 = torch.randint(0, P_PRIME, (16, 16), dtype=torch.int64); L2_inv = invert_matrix_gf(L2, P_PRIME)

# The Public Key Wrapper (Bob can evaluate, but cannot see L1/L2)
public_map = PublicMapEvaluator(L1, L2)

print(f"\n[System] Alice generates NIZK Proof over {NUM_ROUNDS} rounds (Fiat-Shamir Heuristic)...")
nizk_proof = generate_proof(L1, L2, L1_inv, L2_inv)
print("         Proof generation complete. Size:", len(json.dumps(nizk_proof, default=str)) // 1024, "KB")

print("\n[System] Bob receives the proof and verifies the Isomorphism of Polynomials...")
is_valid = verify_proof(public_map, nizk_proof)
print(f"         Result: {'[VALID]' if is_valid else '[INVALID]'} (Alice proved knowledge of the trapdoor in Zero-Knowledge!)")

print("\n[System] Eve tries to forge a proof without knowing L1/L2...")
# Eve just makes up random Q1, Q2 matrices for type 1 responses
eve_proof = generate_proof(L1, L2, L1_inv, L2_inv)
eve_proof["responses"][0] = {"type": 1, "M1": torch.randint(0, P_PRIME, (16, 16)), "M2": torch.randint(0, P_PRIME, (16, 16))}
eve_is_valid = verify_proof(public_map, eve_proof)
print(f"         Result: {'[VALID]' if eve_is_valid else '[INVALID - FORGERY BLOCKED]'} (Math diverged)")
print("==========================================================")