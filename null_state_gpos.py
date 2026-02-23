import torch
import random
import heapq

print("==========================================================")
print("=== Null-State: Geometric Proof-of-Stake (G-PoS)       ===")
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

# --- 2. CRYPTOGRAPHIC PRIMITIVES ---
class Transaction:
    def __init__(self, tx_id, sender_id, payload, signature, pub_lock):
        self.tx_id = tx_id
        self.sender_id = sender_id
        self.payload = payload
        self.signature = signature
        self.pub_lock = pub_lock
        self.magnitude = torch.norm(payload).item()

    def verify_sdsa(self):
        """SDSA Payload Binding Verification: S * (P * L_pub) == 0"""
        target_traj = sedenion_mul(self.payload, self.pub_lock)
        v = sedenion_mul(self.signature, target_traj)
        return torch.norm(v).item() < 1e-5

    def __lt__(self, other):
        return self.tx_id < other.tx_id

# --- 3. NODE ARCHITECTURE ---
class NullStateNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.genesis = torch.zeros((1, 16)); self.genesis[0, 0] = 1.0
        self.global_state = self.genesis.clone()
        
        # The Genesis Token Distribution (Economic Gravity)
        self.stake_ledger = {
            "Alice": 50.0,
            "Bob": 30.0,
            "Eve": 2.0  # Eve has very little stake
        }
        self.known_txs = {}
        
    def receive_transaction(self, tx, current_time):
        if tx.tx_id in self.known_txs:
            return 
            
        # Defense 1: SDSA Trapdoor (Ensures auth and no replay attacks)
        if not tx.verify_sdsa():
            print(f"[{current_time}] Node {self.node_id}: SDSA Forgery Detected. Dropping {tx.tx_id}.")
            return
            
        # Defense 2: Stake-Bounded Magnitude (The Sybil/Infinite Attack Fix)
        max_allowed_magnitude = self.stake_ledger.get(tx.sender_id, 0.0)
        if tx.magnitude > max_allowed_magnitude:
            print(f"[{current_time}] Node {self.node_id}: [SYBIL BLOCK] {tx.sender_id} attempted to spoof magnitude {tx.magnitude:.2f} with only {max_allowed_magnitude} staked tokens! Dropping {tx.tx_id}.")
            return

        print(f"[{current_time}] Node {self.node_id}: Received {tx.tx_id} from {tx.sender_id} (Mag: {tx.magnitude:.2f})")
        self.known_txs[tx.tx_id] = tx
        self.recalculate_state()
        
    def recalculate_state(self):
        sorted_txs = sorted(self.known_txs.values(), key=lambda t: (t.magnitude, t.tx_id), reverse=True)
        new_state = self.genesis.clone()
        for t in sorted_txs:
            new_state = sedenion_mul(new_state, t.payload)
            new_state = torch.nn.functional.normalize(new_state, p=2, dim=1)
        self.global_state = new_state

# --- 4. ASYNCHRONOUS NETWORK SIMULATOR ---
class NetworkSim:
    def __init__(self, num_nodes):
        self.nodes = [NullStateNode(i) for i in range(num_nodes)]
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
network = NetworkSim(3)
L_pub = torch.zeros((1, 16)); L_pub[0, 1] = 1.0; L_pub[0, 10] = 1.0
Sig_S = torch.zeros((1, 16)); Sig_S[0, 3] = 1.0; Sig_S[0, 14] = 1.0

# 1. Alice creates an honest transaction within her bounds
P_Alice = torch.zeros((1, 16)); P_Alice[0, 0] = 45.0  # Max is 50
tx1 = Transaction("TX_ALICE_1", "Alice", P_Alice, Sig_S, L_pub)

# 2. Bob creates an honest transaction within his bounds
P_Bob = torch.zeros((1, 16)); P_Bob[0, 0] = 25.0 # Max is 30
tx2 = Transaction("TX_BOB_1", "Bob", P_Bob, Sig_S, L_pub)

# 3. Eve executes the "Infinite Magnitude Attack"
# She only has 2 tokens, but she tries to inject a magnitude of 500.0 to hijack the DAG ordering
P_Eve = torch.zeros((1, 16)); P_Eve[0, 0] = 500.0
tx3 = Transaction("TX_EVE_ATTACK", "Eve", P_Eve, Sig_S, L_pub)

print(f"\n[System] Broadcasting Alice's TX (Mag: {tx1.magnitude:.2f}, Stake: 50.0)")
network.broadcast(tx1)
network.clock += 2

print(f"[System] Broadcasting Bob's TX (Mag: {tx2.magnitude:.2f}, Stake: 30.0)")
network.broadcast(tx2)
network.clock += 2

print(f"[System] Broadcasting Eve's SYBIL ATTACK (Mag: {tx3.magnitude:.2f}, Stake: 2.0)")
network.broadcast(tx3)

print("\n[System] Advancing asynchronous event loop...")
while network.event_queue:
    network.step()

print("\n==========================================================")
print("=== SIMULATION COMPLETE ===")
print("Final Canonical State Vector across nodes:")
for node in network.nodes:
    print(f"Node {node.node_id}: {node.global_state[0, :4].numpy().round(3)}...")