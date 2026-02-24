use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_dft::Radix2Dit;
use p3_field::AbstractField;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

#[cfg(debug_assertions)]
use p3_uni_stark::DebugConstraintBuilder;

use p3_uni_stark::{
    prove, verify, ProverConstraintFolder, StarkGenericConfig,
    SymbolicAirBuilder, Val, VerifierConstraintFolder,
};
use std::time::Instant;
use p3_commit::ExtensionMmcs;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use p3_uni_stark::StarkConfig;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use sha2::{Sha256, Digest};

// ============================================================================
// OCTONION ALGEBRA OVER F_P
// ============================================================================
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Octonion<F>(pub [F; 8]);

impl<F: AbstractField> Octonion<F> {
    pub fn mul(a: Self, b: Self) -> Self {
        let a = &a.0;
        let b = &b.0;
        let mut r = core::array::from_fn(|_| F::zero());

        r[0] = a[0].clone() * b[0].clone() - a[1].clone() * b[1].clone() - a[2].clone() * b[2].clone() - a[3].clone() * b[3].clone() - a[4].clone() * b[4].clone() - a[5].clone() * b[5].clone() - a[6].clone() * b[6].clone() - a[7].clone() * b[7].clone();
        r[1] = a[0].clone() * b[1].clone() + a[1].clone() * b[0].clone() + a[2].clone() * b[3].clone() - a[3].clone() * b[2].clone() + a[4].clone() * b[5].clone() - a[5].clone() * b[4].clone() - a[6].clone() * b[7].clone() + a[7].clone() * b[6].clone();
        r[2] = a[0].clone() * b[2].clone() - a[1].clone() * b[3].clone() + a[2].clone() * b[0].clone() + a[3].clone() * b[1].clone() + a[4].clone() * b[6].clone() + a[5].clone() * b[7].clone() - a[6].clone() * b[4].clone() - a[7].clone() * b[5].clone();
        r[3] = a[0].clone() * b[3].clone() + a[1].clone() * b[2].clone() - a[2].clone() * b[1].clone() + a[3].clone() * b[0].clone() + a[4].clone() * b[7].clone() - a[5].clone() * b[6].clone() + a[6].clone() * b[5].clone() - a[7].clone() * b[4].clone();
        r[4] = a[0].clone() * b[4].clone() - a[1].clone() * b[5].clone() - a[2].clone() * b[6].clone() - a[3].clone() * b[7].clone() + a[4].clone() * b[0].clone() + a[5].clone() * b[1].clone() + a[6].clone() * b[2].clone() + a[7].clone() * b[3].clone();
        r[5] = a[0].clone() * b[5].clone() + a[1].clone() * b[4].clone() - a[2].clone() * b[7].clone() + a[3].clone() * b[6].clone() - a[4].clone() * b[1].clone() + a[5].clone() * b[0].clone() - a[6].clone() * b[3].clone() + a[7].clone() * b[2].clone();
        r[6] = a[0].clone() * b[6].clone() + a[1].clone() * b[7].clone() + a[2].clone() * b[4].clone() - a[3].clone() * b[5].clone() - a[4].clone() * b[2].clone() + a[5].clone() * b[3].clone() + a[6].clone() * b[0].clone() - a[7].clone() * b[1].clone();
        r[7] = a[0].clone() * b[7].clone() - a[1].clone() * b[6].clone() + a[2].clone() * b[5].clone() + a[3].clone() * b[4].clone() - a[4].clone() * b[3].clone() - a[5].clone() * b[2].clone() + a[6].clone() * b[1].clone() + a[7].clone() * b[0].clone();

        Octonion(r)
    }

    pub fn add(a: Self, b: Self) -> Self {
        let mut r = core::array::from_fn(|_| F::zero());
        for i in 0..8 { r[i] = a.0[i].clone() + b.0[i].clone(); }
        Octonion(r)
    }

    pub fn sub(a: Self, b: Self) -> Self {
        let mut r = core::array::from_fn(|_| F::zero());
        for i in 0..8 { r[i] = a.0[i].clone() - b.0[i].clone(); }
        Octonion(r)
    }

    pub fn conj(a: Self) -> Self {
        let mut r = core::array::from_fn(|_| F::zero());
        r[0] = a.0[0].clone();
        for i in 1..8 { r[i] = F::zero() - a.0[i].clone(); }
        Octonion(r)
    }
}

// ============================================================================
// SEDENION ALGEBRA OVER F_P
// ============================================================================
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sedenion<F>(pub [F; 16]);

impl<F: AbstractField> Sedenion<F> {
    pub fn split(&self) -> (Octonion<F>, Octonion<F>) {
        let mut a = core::array::from_fn(|_| F::zero());
        let mut b = core::array::from_fn(|_| F::zero());
        for i in 0..8 {
            a[i] = self.0[i].clone();
            b[i] = self.0[i+8].clone();
        }
        (Octonion(a), Octonion(b))
    }

    pub fn combine(a: Octonion<F>, b: Octonion<F>) -> Self {
        let mut r = core::array::from_fn(|_| F::zero());
        for i in 0..8 {
            r[i] = a.0[i].clone();
            r[i+8] = b.0[i].clone();
        }
        Sedenion(r)
    }

    pub fn mul(x: Self, y: Self) -> Self {
        let (xa, xb) = x.split();
        let (ya, yb) = y.split();

        let yb_conj = Octonion::conj(yb.clone());
        let ya_conj = Octonion::conj(ya.clone());

        // z.a = x.a * y.a - y.b^* * x.b
        let term1 = Octonion::mul(xa.clone(), ya.clone());
        let term2 = Octonion::mul(yb_conj, xb.clone());
        let za = Octonion::sub(term1, term2);

        // z.b = y.b * x.a + x.b * y.a^*
        let term3 = Octonion::mul(yb, xa);
        let term4 = Octonion::mul(xb, ya_conj);
        let zb = Octonion::add(term3, term4);

        Self::combine(za, zb)
    }
}

// ============================================================================
// S-VDF: SEDENIONIC AIR (Degree 3)
// ============================================================================
#[derive(Clone, Debug)]
pub struct SedenionVdfAir;

impl<F> BaseAir<F> for SedenionVdfAir {
    fn width(&self) -> usize {
        // 16 dimensions for the State (X_t)
        // 16 dimensions for the Generator (G_t)
        32
    }
}

impl<AB: AirBuilder<F = BabyBear> + AirBuilderWithPublicValues> Air<AB> for SedenionVdfAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);
        
        // FIX: Copy public values into a local vector of Expressions to instantly 
        // release the immutable borrow on the builder.
        let pv: Vec<AB::Expr> = builder.public_values().iter().copied().map(Into::into).collect();

        let x_local = Sedenion::<AB::Expr>(core::array::from_fn(|i| local[i].into()));
        let g_local = Sedenion::<AB::Expr>(core::array::from_fn(|i| local[i+16].into()));

        // 1. Boundary Constraints
        // Initial Generator is e0: [1, 0, 0 ... 0]
        builder.when_first_row().assert_eq(local[16], AB::Expr::one());
        for i in 1..16 {
            builder.when_first_row().assert_eq(local[16+i], AB::Expr::zero());
        }

        // Public values map the initial and final states of X
        for i in 0..16 {
            builder.when_first_row().assert_eq(local[i], pv[i].clone());
            builder.when_last_row().assert_eq(local[i], pv[i+16].clone());
        }

        // 2. Generator Transition Constraint (Cyclic Shift Right)
        // This natively rotates the orthogonal generator without a preprocessed trace
        for i in 0..16 {
            let prev_i = if i == 0 { 15 } else { i - 1 };
            builder.when_transition().assert_eq(next[16+i], local[16+prev_i]);
        }

        // 3. Sedenionic VDF Transition Constraint: X_{t+1} = (X_t * G_t) * X_t
        // Inner = X_t * G_t
        let inner = Sedenion::mul(x_local.clone(), g_local);
        // Expected Next = Inner * X_t
        let expected_next = Sedenion::mul(inner, x_local);

        for i in 0..16 {
            builder.when_transition().assert_eq(next[i], expected_next.0[i].clone());
        }
    }
}
// ============================================================================
// PROVER: S-VDF SEQUENTIAL EVALUATOR
// ============================================================================
pub fn run_sedenion_vdf_grind(
    seed: Sedenion<BabyBear>,
    t: usize,
) -> Vec<[BabyBear; 32]> {
    let mut history = Vec::with_capacity(t + 1);
    
    let mut current_x = seed;
    let mut current_g = [BabyBear::zero(); 16];
    current_g[0] = BabyBear::one();

    for _ in 0..t {
        // Record trace row
        let mut row = [BabyBear::zero(); 32];
        for i in 0..16 {
            row[i] = current_x.0[i];
            row[16+i] = current_g[i];
        }
        history.push(row);

        // Advance X_{t+1} = (X_t * G_t) * X_t
        let g_sed = Sedenion(current_g.clone());
        let inner = Sedenion::mul(current_x.clone(), g_sed);
        current_x = Sedenion::mul(inner, current_x);

        // Advance G_{t+1} (Rotate)
        let mut next_g = [BabyBear::zero(); 16];
        for i in 0..16 {
            let prev_i = if i == 0 { 15 } else { i - 1 };
            next_g[i] = current_g[prev_i];
        }
        current_g = next_g;
    }

    // Push final trace row
    let mut row = [BabyBear::zero(); 32];
    for i in 0..16 {
        row[i] = current_x.0[i];
        row[16+i] = current_g[i];
    }
    history.push(row);

    history
}

// ============================================================================
// PRODUCTION STARK ORCHESTRATION
// ============================================================================

#[cfg(debug_assertions)]
pub fn generate_stark_proof<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> p3_uni_stark::Proof<SC>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>
        + Air<SymbolicAirBuilder<Val<SC>>>
        + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>>>,
{
    prove(config, air, challenger, trace, public_values)
}

#[cfg(not(debug_assertions))]
pub fn generate_stark_proof<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> p3_uni_stark::Proof<SC>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>
        + Air<SymbolicAirBuilder<Val<SC>>>,
{
    prove(config, air, challenger, trace, public_values)
}

pub fn verify_stark_proof<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    proof: &p3_uni_stark::Proof<SC>,
    public_values: &Vec<Val<SC>>,
) -> Result<(), p3_uni_stark::VerificationError>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>,
{
    verify(config, air, challenger, proof, public_values)
}

// ============================================================================
// CRYPTOGRAPHIC SEEDING
// ============================================================================
pub fn hash_to_sedenion(seed_string: &str) -> Sedenion<BabyBear> {
    let mut hasher = Sha256::new();
    hasher.update(seed_string.as_bytes());
    let result = hasher.finalize();
    
    let mut state = [BabyBear::zero(); 16];
    for i in 0..16 {
        // Take 2 bytes for each dimension (16-bit integer fits safely in 31-bit BabyBear field)
        let bytes = [result[i * 2], result[i * 2 + 1]];
        let val = u16::from_be_bytes(bytes) as u32;
        state[i] = BabyBear::from_canonical_u32(val);
    }
    
    Sedenion(state)
}

pub fn test_e2e_proof() {
    println!("=================================================================");
    println!("=== S-VDF: Sedenionic STARK Engine (Degree 3) ===");
    println!("=================================================================\n");

    // 1. System Parameters
    #[cfg(debug_assertions)]
    let pow_steps = 12;
    #[cfg(not(debug_assertions))]
    let pow_steps = 22;
    let t_steps = 1 << pow_steps;
    
    // Generate a DENSE 16-dimensional starting state
    let seed_phrase = "Genesis Block Hash 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";
    let seed_vals = hash_to_sedenion(seed_phrase);
    
    println!("[System] Initial Seed Phrase: '{}'", seed_phrase);
    println!("[System] Derived Dense Sedenion [0..3]: {:?}", &seed_vals.0[0..4]);

    // 2. Evaluation Phase
    println!("[Step 1] EVALUATOR: Grinding Non-Associative Chaotic Walk (T={})...", t_steps);
    let start_eval = Instant::now();
    let trace_history = run_sedenion_vdf_grind(seed_vals, t_steps);
    let eval_duration = start_eval.elapsed();

    let final_row = trace_history.last().unwrap();
    println!("   > Evaluation Finished: {:.4}ms", eval_duration.as_secs_f64() * 1000.0);
    println!("   > Final State [0]: {:?}", final_row[0]);

    // 3. Arithmetization Phase
    let mut trace_data = Vec::with_capacity(t_steps * 32);
    for step in trace_history.iter().take(t_steps) {
        trace_data.extend_from_slice(step);
    }
    let trace_matrix = RowMajorMatrix::new(trace_data, 32);

    let initial_state = trace_history[0];
    let final_state = trace_history[t_steps - 1]; 

    // Public Values: 16 vars for initial X, 16 vars for final X
    let mut public_values = Vec::new();
    public_values.extend_from_slice(&initial_state[0..16]);
    public_values.extend_from_slice(&final_state[0..16]);

    // ========================================================================
    // THE CAMERA: PLONKY3 STARK CONFIGURATION
    // ========================================================================
    type Val = BabyBear;
    type Challenge = Val; 

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher32<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type Compress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
    let compress = Compress::new(byte_hash);

    type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);
    
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let dft = Radix2Dit::<Val>::default();

    // The degree is 3, so a blowup of 4 (2^2) is perfectly sufficient
    let fri_config = FriConfig {
        log_blowup: 2, 
        num_queries: 100, 
        proof_of_work_bits: 16, 
        mmcs: challenge_mmcs,
    };
    
    type Pcs = TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(pow_steps, dft, val_mmcs, fri_config);

    type ByteChallenger = HashChallenger<u8, ByteHash, 32>;
    type Challenger = SerializingChallenger32<Val, ByteChallenger>;
    
    let config = StarkConfig::<Pcs, Challenge, Challenger>::new(pcs);

    let air = SedenionVdfAir;

    // 4. Proving Phase
    println!("\n[Step 2] PROVER: Compressing Execution Trace into STARK Proof...");
    let byte_chall_prove = ByteChallenger::new(vec![], byte_hash.clone());
    let mut challenger_prove = Challenger::new(byte_chall_prove);
    let start_prove = Instant::now();

    let proof = generate_stark_proof(&config, &air, &mut challenger_prove, trace_matrix, &public_values);

    let prove_duration = start_prove.elapsed();
    println!("   > Evaluation Finished: {:.4}ms", prove_duration.as_secs_f64() * 1000.0);
    println!("   > S-VDF Receipt Generated Successfully.");

    // 5. Verification Phase
    println!("\n[Step 3] VERIFIER: Validating VDF via Succinct Argument...");
    let byte_chall_verify = ByteChallenger::new(vec![], byte_hash.clone());
    let mut challenger_verify = Challenger::new(byte_chall_verify);
    let start_verify = Instant::now();

    let verification_result = verify_stark_proof(&config, &air, &mut challenger_verify, &proof, &public_values);

    let verify_duration = start_verify.elapsed();
    
    match verification_result {
        Ok(_) => println!("   > Proof VERIFIED. Integrity of time confirmed."),
        Err(e) => println!("   > Proof FAILED: {:?}", e),
    }

    let total_prover_time = eval_duration + prove_duration;
    let speedup = total_prover_time.as_nanos() as f64 / verify_duration.as_nanos().max(1) as f64;

    println!("\n[CONCLUSION]");
    println!("   > Protocol: S-VDF (Sedenionic Chaotic Walk).");
    println!("   > Hardness: Pure Topological Impedance (Degree 3).");
    println!("   > Efficiency: {:.0}x Asymmetric Speedup (Prover vs Verifier).", speedup.round());
    println!("=================================================================\n");
}