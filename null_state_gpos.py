import torch
import hashlib
import heapq
import random

print("==========================================================")
print("=== Null-State: Galois Field GF(p) & VRF Consensus     ===")
print("==========================================================")

# --- 1. DISCRETE GALOIS FIELD ALGEBRA (GF(p)) ---
# Using a Mersenne Prime for the finite field (e.g., 2^31 - 1)
P = 2147483647 

def mod_p(tensor):
    return torch.remainder(tensor, P)

def octonion_mul_gf(a, b):
    c = torch.zeros_like(a, dtype=torch.int64)
    c[:, 0] = a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3] - a[:,4]*b[:,4] - a[:,5]*b[:,5] - a[:,6]*b[:,6] - a[:,7]*b[:,7]
    c[:, 1] = a[:,0]*b[:,1] + a[:,1]*b[:,0] + a[:,2]*b[:,4] + a[:,3]*b[:,7] - a[:,4]*b[:,2] + a[:,5]*b[:,6] - a[:,6]*b[:,5] - a[:,7]*b[:,3]
    c[:, 2] = a[:,0]*b[:,2] - a[:,1]*b[:,4] + a[:,2]*b[:,0] + a[:,3]*b[:,5] + a[:,4]*b[:,1] - a[:,5]*b[:,3] + a[:,6]*b[:,7] - a[:,7]*b[:,6]
    c[:, 3] = a[:,0]*b[:,3] - a[:,1]*b[:,7] - a[:,2]*b[:,5] + a[:,3]*b[:,0] + a[:,4]*b[:,6] + a[:,5]*b[:,2] - a[:,6]*b[:,4] + a[:,7]*b[:,1]
    c[:, 4] = a[:,0]*b[:,4] + a[:,1]*b[:,2] - a[:,2]*b[:,1] - a[:,3]*b[:,6] + a[:,4]*b[:,0] + a[:,5]*b[:,7] + a[:,6]*b[:,3] - a[:,7]*b[:,5]
    c[:, 5] = a[:,0]*b[:,5] - a[:,1]*b[:,6] + a[:,2]*b[:,3] - a[:,3]*b[:,2] - a[:,4]*b[:,7] + a[:,5]*b[:,0] + a[:,6]*b[:,1] + a[:,7]*b[:,4]
    c[:, 6] = a[:,0]*b[:,6] + a[:,1]*b[:,5] - a[:,2]*b[:,7] + a[:,3]*b[:,4] - a[:,4]*b[:,3] - a[:,5]*b[:,1] + a[:,6]*b[:,0] + a[:,7]*b[:,2]
    c[:, 7] = a[:,0]*b[:,7] + a[:,1]*b[:,3] + a[:,2]*b[:,6] - a[:,3]*b[:,1] + a[:,4]*b[:,5] - a[:,5]*b[:,4] - a[:,6]*b[:,2] + a[:,7]*b[:,0]
    return mod_p(c)

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

# Protocol Constant C for Affine Translation (Prevents Ledger Annihilation)
C_TRANS = torch.ones((1, 16), dtype=torch.int64) * 999 

# --- 2. CRYPTOGRAPHIC PRIMITIVES ---
class Transaction:
    def __init__(self, tx_id, sender_id, payload, signature, pub_lock, anchor_state):
        self.tx_id = tx_id
        self.sender_id = sender_id
        self.payload = mod_p(payload)
        self.signature = mod_p(signature)
        self.pub_lock = mod_p(pub_lock)
        self.anchor_state = anchor_state

    def verify_sdsa(self):
        """Discrete Verification: S * (P * L_pub) == 0 mod P"""
        target_traj = sedenion_mul_gf(self.payload, self.pub_lock)
        v = sedenion_mul_gf(self.signature, target_traj)
        return torch.sum(v).item() == 0

    def __lt__(self, other):
        return self.tx_id < other.tx_id

# --- 3. NODE ARCHITECTURE ---
class NullStateNode_GF:
    def __init__(self, node_id):
        self.node_id = node_id
        self.genesis = torch.zeros((1, 16), dtype=torch.int64); self.genesis[0, 0] = 1
        self.global_state = self.genesis.clone()
        self.known_txs = {}
        
    def generate_vrf_anchor(self):
        """Generates deterministic pseudo-random vector from current state for unbiased sorting"""
        state_bytes = self.genesis.numpy().tobytes()
        h = hashlib.sha256(state_bytes).digest()
        # Map hash bytes to a 16D integer vector
        r_anchor = torch.tensor([int.from_bytes(h[i:i+2], 'big') for i in range(0, 32, 2)], dtype=torch.int64)
        return mod_p(r_anchor.unsqueeze(0))

    def receive_transaction(self, tx, current_time):
        if tx.tx_id in self.known_txs:
            return 
            
        if not tx.verify_sdsa():
            print(f"[{current_time}] Node {self.node_id}: SDSA Forgery Detected. Dropping {tx.tx_id}.")
            return

        print(f"[{current_time}] Node {self.node_id}: Received {tx.tx_id} from {tx.sender_id}")
        self.known_txs[tx.tx_id] = tx
        self.recalculate_state()
        
    def recalculate_state(self):
        R_anchor = self.generate_vrf_anchor()
        
        # VRF Sorting: (P dot R_anchor) mod P
        def vrf_score(tx):
            return mod_p(torch.sum(tx.payload * R_anchor)).item()
            
        sorted_txs = sorted(self.known_txs.values(), key=lambda t: (vrf_score(t), t.tx_id), reverse=True)
        
        new_state = self.genesis.clone()
        for t in sorted_txs:
            # Affine State Translation: G_t = (G_{t-1} * P_tx) + C mod P
            mult_res = sedenion_mul_gf(new_state, t.payload)
            new_state = mod_p(mult_res + C_TRANS)
            
        self.global_state = new_state

# --- 4. NETWORK SIMULATOR ---
class NetworkSim_GF:
    def __init__(self, num_nodes):
        self.nodes = [NullStateNode_GF(i) for i in range(num_nodes)]
        self.clock = 0
        self.event_queue = []
        
    def broadcast(self, tx):
        for node in self.nodes:
            delay = random.randint(5, 50)
            heapq.heappush(self.event_queue, (self.clock + delay, node.node_id, tx))
            
    def step(self):
        delivered = False
        while self.event_queue and self.event_queue[0][0] <= self.clock:
            delivery_time, node_id, tx = heapq.heappop(self.event_queue)
            self.nodes[node_id].receive_transaction(tx, self.clock)
            delivered = True
        self.clock += 1
        return delivered

# --- 5. RUN THE SIMULATION ---
network = NetworkSim_GF(3)

# Setup keys (using integers now)
L_pub = torch.zeros((1, 16), dtype=torch.int64); L_pub[0, 1] = 1; L_pub[0, 10] = 1
Sig_S = torch.zeros((1, 16), dtype=torch.int64); Sig_S[0, 3] = 1; Sig_S[0, 14] = 1

P_Alice = torch.zeros((1, 16), dtype=torch.int64); P_Alice[0, 0] = 45
tx1 = Transaction("TX_ALICE_1", "Alice", P_Alice, Sig_S, L_pub, network.nodes[0].genesis)

P_Bob = torch.zeros((1, 16), dtype=torch.int64); P_Bob[0, 0] = 25 
tx2 = Transaction("TX_BOB_1", "Bob", P_Bob, Sig_S, L_pub, network.nodes[0].genesis)

print("\n[System] Broadcasting Alice and Bob's TXs asynchronously...")
network.broadcast(tx1)
network.broadcast(tx2)

while network.event_queue:
    network.step()

print("\n==========================================================")
print("=== SIMULATION COMPLETE ===")
print("Final Canonical State Vector across nodes (GF(p)):")
for node in network.nodes:
    print(f"Node {node.node_id}: {node.global_state[0, :6].numpy()}... (Truncated)")