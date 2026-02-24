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

/// An Octonion represented by 8 elements in a Field.
/// This structure acts as the state variable for the VDF.
/// Zn+1 = Zn^2 + C + [Zn, C, H(Zn)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Octonion<F>(pub [F; 8]);

impl<F: AbstractField> Octonion<F> {
    /// Non-associative multiplication over the Fano Plane.
    /// This is the primary serial bottleneck that ensures sequential hardness.
    pub fn mul(a: Self, b: Self) -> Self {
        let a = &a.0;
        let b = &b.0;
        let mut r = core::array::from_fn(|_| F::zero());

        // Fano Plane Multiplication Table Logic (Hardened against regressions)
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
        for i in 0..8 {
            r[i] = a.0[i].clone() + b.0[i].clone();
        }
        Octonion(r)
    }

    pub fn sub(a: Self, b: Self) -> Self {
        let mut r = core::array::from_fn(|_| F::zero());
        for i in 0..8 {
            r[i] = a.0[i].clone() - b.0[i].clone();
        }
        Octonion(r)
    }

    /// The Associator measures the failure of the associative law.
    /// [A, B, D] = (AB)D - A(BD). This multilinear map provides Topological Impedance.
    pub fn associator(a: Self, b: Self, d: Self) -> Self {
        let ab_d = Self::mul(Self::mul(a.clone(), b.clone()), d.clone());
        let a_bd = Self::mul(a, Self::mul(b, d));
        Self::sub(ab_d, a_bd)
    }
}

/// OctoStarkAir: The production-grade AIR for the VDF.
/// Defines the polynomial constraints for non-associative sequential delay.
#[derive(Clone, Debug)]
pub struct OctoStarkAir {
    pub c: Octonion<BabyBear>, // The public Cosmological Constant
}

impl<F> BaseAir<F> for OctoStarkAir {
    fn width(&self) -> usize {
        8
    }
}

impl<AB: AirBuilder<F = BabyBear> + AirBuilderWithPublicValues> Air<AB> for OctoStarkAir {
    fn eval(&self, builder: &mut AB) {
        // Extract handles from the builder first to release the immutable borrow.
        // AB::Var handles are copyable indices into the trace.
        let local: [AB::Var; 8] = {
            let main = builder.main();
            let slice = main.row_slice(0);
            core::array::from_fn(|i| slice[i])
        };
        let next: [AB::Var; 8] = {
            let main = builder.main();
            let slice = main.row_slice(1);
            core::array::from_fn(|i| slice[i])
        };
        let public_values: [AB::PublicVar; 16] = {
            let pv = builder.public_values();
            core::array::from_fn(|i| pv[i])
        };

        // 1. Boundary Constraints: Genesis Seed and Final Attractor
        // PV mapping: [0..8] is the initial state, [8..16] is the final state.
        for i in 0..8 {
            builder.when_first_row().assert_eq(local[i], public_values[i]);
            builder
                .when_last_row()
                .assert_eq(local[i], public_values[i + 8]);
        }

        // 2. Transition Constraints: The Hourglass Step
        // Mapping Var handles to Expr elements for algebraic manipulation.
        let z_local = Octonion::<AB::Expr>(core::array::from_fn(|i| local[i].into()));
        let c_expr = Octonion::<AB::Expr>(core::array::from_fn(|i| self.c.0[i].into()));

        // Algebraic Hash H(Zn) injected as a 3rd generator to bypass Artin's Theorem.
        let h_z_vals = core::array::from_fn(|i| {
            let x = z_local.0[i].clone();
            let x2 = x.clone() * x.clone();
            let x4 = x2.clone() * x2.clone();
            x4 * x2 * x
        });
        let h_z = Octonion(h_z_vals);

        // Zn+1 = Zn^2 + C + [Zn, C, H(Zn)]
        let z_sq = Octonion::mul(z_local.clone(), z_local.clone());
        let assoc = Octonion::associator(z_local, c_expr.clone(), h_z);

        let expected_next = Octonion::add(Octonion::add(z_sq, c_expr), assoc);

        for i in 0..8 {
            builder
                .when_transition()
                .assert_eq(next[i], expected_next.0[i].clone());
        }
    }
}

/// Prover: Sequential evaluation of the non-associative hourglass trace.
pub fn run_vdf_grind(
    seed: Octonion<BabyBear>,
    c: Octonion<BabyBear>,
    t: usize,
) -> Vec<Octonion<BabyBear>> {
    let mut history = Vec::with_capacity(t + 1);
    let mut current = seed;
    for _ in 0..t {
        history.push(current);

        let h_z_vals = core::array::from_fn(|i| {
            let x = current.0[i];
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 * x2 * x
        });
        let h_z = Octonion(h_z_vals);

        let z_sq = Octonion::mul(current, current);
        let assoc = Octonion::associator(current, c, h_z);
        current = Octonion::add(Octonion::add(z_sq, c), assoc);
    }
    history.push(current);
    history
}

// ============================================================================
// PRODUCTION STARK ORCHESTRATION
// ============================================================================

/// Generates a production-grade zk-STARK proof for the OctoSTARK hourglass.
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

// -----------------------------------------------------------------
// RELEASE MODE: Stripped down for absolute maximum speed
// -----------------------------------------------------------------
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

/// Verifies a zk-STARK proof asymmetrically in O(log^2 T) time.
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

pub fn test_e2e_proof() {
    println!("=================================================================");
    println!("=== Sedenion VDF: Production STARK Engine ===");
    println!("=================================================================\n");

    // 1. System Parameters
    #[cfg(debug_assertions)]
    let pow_steps = 16;
    #[cfg(not(debug_assertions))]
    let pow_steps = 22;
    let t_steps = 1 << pow_steps;
    // Power of 2 required for optimal DFT/FRI
    let seed_vals = Octonion([BabyBear::from_canonical_u32(7); 8]);
    let c_vals = Octonion([BabyBear::from_canonical_u32(1337); 8]);

    // 2. Evaluation Phase (Sequential Bottleneck)
    println!("[Step 1] EVALUATOR: Grinding Non-Associative Hourglass (T={})...", t_steps);
    let start_eval = Instant::now();
    let trace_history = run_vdf_grind(seed_vals, c_vals, t_steps);
    let eval_duration = start_eval.elapsed();

    let final_state = *trace_history.last().unwrap();
    println!("   > Evaluation Finished: {:.4}ms", eval_duration.as_secs_f64() * 1000.0);
    println!("   > Final State [0]: {:?}", final_state.0[0]);

    // 3. Arithmetization Phase
    // We must strictly enforce a power-of-two row count for the FFT!
    // We take exactly `t_steps` (1024) rows from our history.
    let mut trace_data = Vec::with_capacity(t_steps * 8);
    for step in trace_history.iter().take(t_steps) {
        trace_data.extend_from_slice(&step.0);
    }
    let trace_matrix = RowMajorMatrix::new(trace_data, 8);

    // Our public values must match the exact start and end of this matrix.
    let initial_state = trace_history[0];
    let final_state = trace_history[t_steps - 1]; 

    // Prepare Public Values (Dynamic boundary constraints)
    let mut public_values = Vec::new();
    public_values.extend_from_slice(&initial_state.0);
    public_values.extend_from_slice(&final_state.0);

    // ========================================================================
    // THE CAMERA: PLONKY3 STARK CONFIGURATION
    // ========================================================================
    type Val = BabyBear;
    type Challenge = Val; // For simplicity in this prototype, we use the base field.

    // Instantiate the Hash Function (The Mixer)
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher32<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    // Instantiate the Compression Function for the Merkle Tree
    type Compress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
    let compress = Compress::new(byte_hash);

    // Instantiate the Merkle Tree MMCS (The Matrix Commitment)
    type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);
    
    // We use the same MMCS for the challenge extension field 
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let dft = Radix2Dit::<Val>::default();

    // Configure the FRI Protocol (The Logarithmic Folder)
    let fri_config = FriConfig {
        log_blowup: 4, // Reed-Solomon expansion factor
        num_queries: 100, // Number of random checks (determines security level)
        proof_of_work_bits: 16, // Grinding for extra security
        mmcs: challenge_mmcs,
    };

    
    // Tie it all together into the Polynomial Commitment Scheme (PCS)
    type Pcs = TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(pow_steps, dft, val_mmcs, fri_config);

    // The final Stark Configuration
    type ByteChallenger = HashChallenger<u8, ByteHash, 32>;
    type Challenger = SerializingChallenger32<Val, ByteChallenger>;
    
    let config = StarkConfig::<Pcs, Challenge, Challenger>::new(pcs);

    let air = OctoStarkAir { c: c_vals };

    // 4. Proving Phase (The zk-Argument)
    println!("\n[Step 2] PROVER: Compressing Hourglass Trace into STARK Proof...");
    let byte_chall_prove = ByteChallenger::new(vec![], byte_hash.clone());
    let mut challenger_prove = Challenger::new(byte_chall_prove);
    let start_prove = Instant::now();

    // Fire the camera!
    let proof = generate_stark_proof(&config, &air, &mut challenger_prove, trace_matrix, &public_values);

    let prove_duration = start_prove.elapsed();
    println!("   > Evaluation Finished: {:.4}ms", prove_duration.as_secs_f64() * 1000.0);
    println!("   > STARK Receipt Generated Successfully.");

    // 5. Verification Phase (Logarithmic Time)
    println!("\n[Step 3] VERIFIER: Validating VDF via Succinct Argument...");
    let byte_chall_verify = ByteChallenger::new(vec![], byte_hash.clone());
    let mut challenger_verify = Challenger::new(byte_chall_verify);
    let start_verify = Instant::now();

    // Check the receipt!
    let verification_result = verify_stark_proof(&config, &air, &mut challenger_verify, &proof, &public_values);

    let verify_duration = start_verify.elapsed();
    
    match verification_result {
        Ok(_) => println!("   > Proof VERIFIED. Integrity of time confirmed."),
        Err(e) => println!("   > Proof FAILED: {:?}", e),
    }

    // 6. Asymmetric Analysis
    let total_prover_time = eval_duration + prove_duration;
    let speedup = total_prover_time.as_nanos() as f64 / verify_duration.as_nanos().max(1) as f64;

    println!("\n[CONCLUSION]");
    println!("   > Protocol: OctoSTARK (APH Framework).");
    println!("   > Hardness: Non-Associative Octonion Associator Gap.");
    println!("   > Efficiency: {:.0}x Asymmetric Speedup (Prover vs Verifier).", speedup.round());
    println!("=================================================================\n");
}
