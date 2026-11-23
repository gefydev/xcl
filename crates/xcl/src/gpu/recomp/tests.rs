//! Instruction semantics checked on the real GPU: small RDNA2 programs are assembled,
//! recompiled to HLSL, compiled with DXC and run as kernels. Skipped (with a note) when
//! there is no D3D12 GPU with SM 6.6 or no DXC; the tests that need wave64 also without it,
//! the others run as two wave32 halves on such an adapter (`gpu_ready_wave32`).

use super::runner::{self, KernelState};
use super::{translate, Mode, Options};
use crate::gpu::rdna2::encode;

fn gpu_ready() -> bool {
    match (runner::adapter(), runner::wave64_supported(), super::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => true,
        (a, w, d) => {
            println!("GPU/DXC wave64 unavailable – skipping ({:?} / {:?} / {:?})", a.as_ref().err(), w.ok(), d.as_ref().err());
            runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
            false
        }
    }
}

fn gpu_ready_wave32() -> bool {
    match (runner::adapter(), super::dxc::location()) {
        (Ok(_), Ok(_)) => true,
        (a, d) => {
            println!("GPU/DXC unavailable – skipping ({:?} / {:?})", a.as_ref().err(), d.as_ref().err());
            runner::skip_test(if a.is_err() { "no GPU" } else { "no DXC" });
            false
        }
    }
}

/// `runner::mesh_draw` needs mesh shaders (not every adapter with wave64 has them).
fn mesh_ready() -> bool {
    match runner::mesh_shaders_supported() {
        Ok(true) => true,
        r => {
            println!("GPU without mesh shaders – skipping ({r:?})");
            runner::skip_test(if r.is_err() { "no GPU" } else { "no mesh shaders" });
            false
        }
    }
}

/// Assemble a program from (name, encoding, fields, literal).
fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, f, l) in list {
        v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    v
}

const V: u64 = 256; // VALU source code of v0

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn geometry_wave32_rejects_cross_half_operations() {
    let one = |name: &str, enc: &str, fields: &[(&str, u64)]| encode(name, enc, fields, None).unwrap();

    let lane = one("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)]);
    assert!(super::geometry_wave32_hazards(&lane).contains("cross-half lane exchange"));

    let branch = one("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)]);
    assert!(super::geometry_wave32_hazards(&branch).contains("branch on a whole-wave mask"));

    let reduce = one("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]);
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));

    let compare = one("S_CMP_EQ_U64", "ENC_SOPC", &[("SSRC0", 126), ("SSRC1", 106)]);
    assert!(super::geometry_wave32_hazards(&compare).contains("scalar condition from a whole-wave mask"));

    let mut scalar = one("S_AND_B64", "ENC_SOP2", &[("SDST", 8), ("SSRC0", 126), ("SSRC1", 106)]);
    scalar.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    assert!(super::geometry_wave32_hazards(&scalar).contains("branch on a half-local scalar condition"));

    // a prefix count over EXEC, VCC or a v_cmp result in an SGPR pair; the lane index is fine
    for mask in [126, 107, 10] {
        let count = one("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", mask), ("SRC1", 128)]);
        assert!(super::geometry_wave32_hazards(&count).contains("prefix count over a whole-wave mask"), "SRC0 {mask}");
    }
    let mut lane = one("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", 128)]);
    lane.extend(one("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", 193), ("SRC1", V + 1)]));
    assert!(super::geometry_wave32_hazards(&lane).is_empty());

    // a v_cmp into s[0:1]: a mask, its 32-bit words too
    let cmp = one("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 168), ("SRC1", V)]);
    let mut reduce = cmp.clone();
    reduce.extend(one("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 0)]));
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));
    let mut reduce = one("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)]);
    reduce.extend(one("S_FLBIT_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 127)]));
    assert!(super::geometry_wave32_hazards(&reduce).contains("whole-wave mask reduction"));
    let mut branch = cmp.clone();
    branch.extend(one("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)]));
    branch.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    assert!(super::geometry_wave32_hazards(&branch).contains("branch on a half-local scalar condition"));
    // the same scalar work on a non-mask SGPR stays accepted
    let mut plain = one("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)]);
    plain.extend(one("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 128)]));
    plain.extend(one("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0)]));
    plain.extend(one("S_ENDPGM", "ENC_SOPP", &[]));
    assert!(super::geometry_wave32_hazards(&plain).is_empty());
}

#[test]
fn geometry_wave_path_selection_preserves_native_wave64() {
    assert!(super::pso::geometry_uses_wave32(32, 32));
    assert!(!super::pso::geometry_uses_wave32(32, 64));
    assert!(!super::pso::geometry_uses_wave32(64, 64));
}

#[test]
fn hlsl_is_generated_for_straight_line_and_branches() {
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)], None),
        ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 133)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert_eq!(t.blocks, 3, "{}", t.hlsl);
    // forward branches only: a plain `if`, no loop and no block dispatch on pc
    assert!(t.hlsl.contains("if (!(") && !t.hlsl.contains("pc ==") && !t.hlsl.contains("for (;;)") && t.hlsl.contains("[WaveSize(64)]"), "{}", t.hlsl);
    // the guarded block chain (fallback form) for the same program
    // (an option, not the environment variable: other tests translate at the same time)
    let g = translate(&code, &Options { structured: false, ..Options::default() }).unwrap();
    assert!(g.hlsl.contains("if (pc == 0u)") && !g.hlsl.contains("while"), "{}", g.hlsl);
    // a backward branch: only its range becomes a loop
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)], None),
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 129)], None),
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 0), ("SSRC1", 128)], None),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None), // back to s_sub
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    // only the range of the backward branch loops; its condition is the loop's exit test
    assert!(t.hlsl.contains("[loop] for (;;)") && t.hlsl.contains("break;") && !t.hlsl.contains("pc =="), "{}", t.hlsl);
    let t = translate(&prog(&[("V_ADD_F32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 2), ("VSRC1", 3)], None)]), &Options::default()).unwrap();
    assert!(!t.hlsl.contains("pc =="), "{}", t.hlsl);
    // unknown to the recompiler: reported, HLSL still produced
    let t = translate(&prog(&[("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 0), ("VSRC", 0), ("ATTR", 0)], None)]), &Options::default()).unwrap();
    assert_eq!(t.unsupported.get("v_interp_p1_f32"), Some(&1));
    let _ = Mode::Kernel;
}

#[test]
fn scalar_bfe_64_handles_unsigned_signed_and_zero_width() {
    let code = prog(&[
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 4), ("SSRC1", 8)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 4), ("SSRC1", 9)], None),
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 4), ("SSRC1", 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("& (0xFFFFFFFFFFFFFFFFull >> (64u -") && t.hlsl.contains("(int64_t)(t") && t.hlsl.contains("== 0u) ? 0ull"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_bitcmp_64_selects_the_indexed_half() {
    let code = prog(&[
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 8)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 4), ("SSRC1", 9)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("select((s8 & 32u) == 0u, s4, s5)") && t.hlsl.contains("select((s9 & 32u) == 0u, s4, s5)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("scc = (uint)(") && t.hlsl.contains("== 0u);"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_64_bit_count_scan_and_set_cluster() {
    let code = prog(&[
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 4)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 4)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 10)], None),
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 11)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("64u - countbits(s4) - countbits(s5)") && t.hlsl.contains("firstbitlow(~s4)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("31u - firstbithigh(s5)") && t.hlsl.contains("s7 | select((s10 & 32u) != 0u") && t.hlsl.contains("s6 & ~select((s11 & 32u) == 0u"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_signed_leading_bit_scans() {
    let code = prog(&[
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 4)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 6)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("ffbh_i(s4)"), "{}", t.hlsl);
    // i64: normalize on the 64-bit sign, then scan the high half first and the low half second
    assert!(t.hlsl.contains("select((int)s7 < 0, ~s6, s6)") && t.hlsl.contains("select((int)s7 < 0, ~s7, s7)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("31u - firstbithigh(") && t.hlsl.contains("63u - firstbithigh(") && t.hlsl.contains("0xFFFFFFFFu"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn scalar_relative_move_cluster() {
    let code = prog(&[
        ("S_MOVRELS_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)], None),
        ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 10)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 10)], None),
        ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 10)], None),
        ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("switch (m0)") && t.hlsl.contains("switch ((m0 & 0x3FFu))") && t.hlsl.contains("switch (((m0 >> 16) & 0x3FFu))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_scalar_relative_moves() {
    if !gpu_ready_wave32() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 130)], None),
        ("S_MOVRELS_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 10)], None),
        ("S_MOVRELS_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 10)], None),
        ("S_MOVRELD_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 10)], None),
        ("S_MOVRELD_B64", "ENC_SOP1", &[("SDST", 24), ("SSRC0", 10)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 255)], Some(0x0003_0001)),
        ("S_MOVRELSD_2_B32", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 10)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[10..14].copy_from_slice(&[100, 101, 102, 103]);
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!((r.sgpr[0], r.sgpr[2], r.sgpr[3]), (102, 102, 103), "relative scalar reads");
    assert_eq!((r.sgpr[22], r.sgpr[26], r.sgpr[27]), (100, 100, 101), "relative scalar writes");
    assert_eq!(r.sgpr[33], 101, "split source/destination offsets");
}

/// v_swaprel_b32: both sides are read-write (fixed VDST + the VGPR at SRC0 + M0); emitted
/// as the same bounded switch over M0 as v_movrel, both old values captured before writes.
#[test]
fn v_swaprel_b32_uses_the_m0_relative_switch() {
    let code = prog(&[("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None)]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // old VDST captured; the M0-relative register read into another temporary
    assert!(t.hlsl.contains("uint t1 = v5;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint t2 = 0u; switch (m0) { case 0u: t2 = v10; break; default: break; }"), "{}", t.hlsl);
    // EXEC-gated exchange of the two captured values
    assert!(t.hlsl.contains("if (lanebit(exec_lo, exec_hi)) { v5 = t2; switch (m0) { case 0u: v10 = t1; break; default: break; } }"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_v_swaprel_b32() {
    if !gpu_ready_wave32() {
        return;
    }
    let p = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 10)], None), // v5 <-> v13
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 30)], None), // exec = s[30:31]: even lanes
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 11)], None), // v6 <-> v14 (active only)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 32)], None), // exec = s[32:33]: all lanes
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 129)], None), // m0 = 1
        ("V_SWAPREL_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 6)], None), // v7 <-> v7: no-op
        // relative targets lie within the program's VGPRs (as in gpu_v_movrel): v15 used directly
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 15)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[30] = 0x5555_5555;
    st.sgpr[31] = 0x5555_5555;
    st.sgpr[32] = 0xFFFF_FFFF;
    st.sgpr[33] = 0xFFFF_FFFF;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[5] = 1000 + l as u32;
        r[13] = 2000 + l as u32;
        r[6] = 3000 + l as u32;
        r[14] = 4000 + l as u32;
        r[7] = 700 + l as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!((v[5], v[13]), (2000 + l as u32, 1000 + l as u32), "lane {l} ordinary swap");
        let expected = if l % 2 == 0 { (4000 + l as u32, 3000 + l as u32) } else { (3000 + l as u32, 4000 + l as u32) };
        assert_eq!((v[6], v[14]), expected, "lane {l} EXEC-gated swap");
        assert_eq!(v[7], 700 + l as u32, "lane {l} overlapping swap is a no-op");
    }
    assert_eq!((r.sgpr[126], r.sgpr[127]), (0xFFFF_FFFF, 0xFFFF_FFFF), "exec restored");
}

#[test]
fn atomic_inc_dec_x2_encoding_families() {
    let code = prog(&[
        ("BUFFER_ATOMIC_INC_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 0), ("SOFFSET", 128)], None),
        ("BUFFER_ATOMIC_DEC_X2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 8), ("SRSRC", 0), ("SOFFSET", 128)], None),
        ("FLAT_ATOMIC_INC_X2", "ENC_FLAT", &[("VDST", 10), ("ADDR", 12), ("DATA", 14)], None),
        ("FLAT_ATOMIC_DEC_X2", "ENC_FLAT", &[("VDST", 16), ("ADDR", 12), ("DATA", 18)], None),
        ("GLOBAL_ATOMIC_INC_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2)], None),
        ("GLOBAL_ATOMIC_DEC_X2", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 22), ("DATA", 28), ("SADDR", 2)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("InterlockedCompareExchange64").count(), 6, "{}", t.hlsl);
    assert!(t.hlsl.contains(") ? (uint64_t)0 :") && t.hlsl.contains("== (uint64_t)0 ||"), "{}", t.hlsl);
    // memory above 4 GiB: every access of the CAS loop (its first read too) goes through the
    // window accessors
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert!(!w.hlsl.contains("mem.Load<") && w.hlsl.matches("= xcl_mLoad2(").count() >= 6, "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}
{}", w.hlsl));
    }
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

/// *_atomic_fcmpswap_x2 (buffer / flat / global): a 64-bit float compare-swap; the payload
/// compares bitwise, exactly like the 32-bit fcmpswap – one InterlockedCompareExchange64.
/// Deterministic emitted text on all three encoding families.
#[test]
fn atomic_fcmpswap_x2_is_a_bitwise_64_bit_cas() {
    let code = prog(&[
        ("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)], None),
        ("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 12), ("ADDR", 14), ("DATA", 16), ("GLC", 1)], None),
        ("GLOBAL_ATOMIC_FCMPSWAP_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // one native 64-bit CAS per instruction – no emulation loop
    assert_eq!(t.hlsl.matches("mem.InterlockedCompareExchange64(").count(), 3, "{}", t.hlsl);
    // [DATA:DATA+1] is the swap value, [DATA+2:DATA+3] the compare value (128-bit operand)
    assert!(t.hlsl.contains("_v = (uint64_t)v4 | ((uint64_t)v5 << 32); uint64_t") && t.hlsl.contains("_cmp = (uint64_t)v6 | ((uint64_t)v7 << 32);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_cmp = (uint64_t)v18 | ((uint64_t)v19 << 32);"), "{}", t.hlsl);
    assert!(t.hlsl.contains("_cmp = (uint64_t)v26 | ((uint64_t)v27 << 32);"), "{}", t.hlsl);
    // 8-byte alignment of all three cells; the buffer form's bounds check reads records in dwords
    assert_eq!(t.hlsl.matches("& ~7u;").count(), 3, "{}", t.hlsl);
    assert!(t.hlsl.contains("_off + 8u <= s10"), "{}", t.hlsl);
    // GLC returns the old u64 as a low / high dword pair – in place in VDATA for the buffer
    // form, into VDST for flat / global
    assert_eq!(t.hlsl.matches("_old >> 32").count(), 3, "{}", t.hlsl);
    assert!(t.hlsl.contains("v4 = (uint)t") && t.hlsl.contains("v5 = (uint)(t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v12 = (uint)t") && t.hlsl.contains("v13 = (uint)(t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v20 = (uint)t") && t.hlsl.contains("v21 = (uint)(t"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

/// *_atomic_fmin_x2 / fmax_x2 (buffer / flat / global): f64 min / max as one native 64-bit
/// integer atomic picked by the operand's sign, like the f32 forms – no CAS loop.
#[test]
fn atomic_fmin_fmax_x2_are_native_64_bit_atomics() {
    let code = prog(&[
        ("BUFFER_ATOMIC_FMIN_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)], None),
        ("BUFFER_ATOMIC_FMAX_X2", "ENC_MUBUF", &[("VDATA", 6), ("VADDR", 8), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("FLAT_ATOMIC_FMIN_X2", "ENC_FLAT", &[("VDST", 10), ("ADDR", 12), ("DATA", 14), ("GLC", 1)], None),
        ("FLAT_ATOMIC_FMAX_X2", "ENC_FLAT", &[("VDST", 16), ("ADDR", 12), ("DATA", 18)], None),
        ("GLOBAL_ATOMIC_FMIN_X2", "ENC_FLAT_GLBL", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)], None),
        ("GLOBAL_ATOMIC_FMAX_X2", "ENC_FLAT_GLBL", &[("VDST", 26), ("ADDR", 22), ("DATA", 28), ("SADDR", 2)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(!t.hlsl.contains("InterlockedCompareExchange64"), "{}", t.hlsl);
    // one signed and one unsigned branch per instruction, chosen at +inf's bits
    assert_eq!(t.hlsl.matches("<= 0x7FF0000000000000ull)").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("mem.InterlockedMin64(").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("mem.InterlockedMax64(").count(), 6, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("& ~7u;").count(), 6, "{}", t.hlsl);
    // memory above 4 GiB: the program's atomics go through the window accessors (whose own
    // bodies call mem / mem1)
    let w = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert_eq!(w.hlsl.matches("xcl_mInterlockedMin64(t").count(), 6, "{}", w.hlsl);
    assert_eq!(w.hlsl.matches("xcl_mInterlockedMax64(t").count(), 6, "{}", w.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
        super::dxc::compile(&w.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", w.hlsl));
    }
}

#[test]
fn unary_f64_math_vop1_and_vop3() {
    let code = prog(&[
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 10)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 10)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 10)], None),
        ("V_CEIL_F64", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 10), ("NEG", 1)], None),
        ("V_FRACT_F64", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 10), ("ABS", 1)], None),
        ("V_SQRT_F64", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 10)], None),
        ("V_RSQ_F64", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 10)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("ceil_d(").count(), 3, "{}", t.hlsl); // helper plus two calls
    assert_eq!(t.hlsl.matches("fract_d(").count(), 3, "{}", t.hlsl); // helper plus two calls
    // f64 roots through sqrt_d (HLSL's sqrt is f32): V_SQRT / V_RSQ, twice each
    assert_eq!(t.hlsl.matches("sqrt_d(").count(), 1 + 4, "{}", t.hlsl); // helper plus four calls
    assert_eq!(t.hlsl.matches("(double)1 / sqrt_d(").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn f64_class_compare_vopc_and_vop3() {
    let code = prog(&[
        ("V_CMP_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 10), ("VSRC1", 12)], None),
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 10), ("SRC1", V + 12), ("ABS", 1)], None),
        ("V_CMPX_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 10), ("VSRC1", 13)], None),
        ("V_CMPX_CLASS_F64", "ENC_VOP3", &[("VDST", 126), ("SRC0", V + 10), ("SRC1", V + 13), ("NEG", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert_eq!(t.hlsl.matches("fclass64(").count(), 5, "{}", t.hlsl); // helper plus four calls
    // SRC0 is the 64-bit f64 operand (VGPR/SGPR pair), SRC1 stays a plain 32-bit mask;
    // abs/neg act on the double operand only
    assert!(t.hlsl.contains("(v12) >> fclass64(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v12) >> fclass64(asdouble(v10, (v11 & 0x7FFFFFFFu)))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v13) >> fclass64(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    // mask destinations go through the shared compare/write-mask path: vcc, an SGPR pair, exec
    assert!(t.hlsl.contains("vcc_lo = t") && t.hlsl.contains("vcc_hi = t"), "{}", t.hlsl);
    assert!(t.hlsl.contains("s20 = t") && t.hlsl.contains("s21 = t"), "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("exec_lo = t").count(), 2, "{}", t.hlsl);
    assert_eq!(t.hlsl.matches("exec_hi = t").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_f64_class_compare() {
    if !gpu_ready_wave32() {
        return;
    }
    let p = prog(&[
        ("V_CMP_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 50), ("VSRC1", 52)], None),                        // vcc
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 50), ("SRC1", V + 52)], None),      // s[20:21]
        ("V_CMP_CLASS_F64", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 50), ("SRC1", 8), ("ABS", 1)], None), // |x|, mask in s8
        ("V_CMPX_CLASS_F64", "ENC_VOPC", &[("SRC0", V + 50), ("VSRC1", 53)], None),                       // exec
    ]);
    let classes = [
        0x7FF0_0000_0000_0001u64,
        0x7FF8_0000_0000_0000,
        f64::NEG_INFINITY.to_bits(),
        (-1.0f64).to_bits(),
        0x8000_0000_0000_0001,
        (-0.0f64).to_bits(),
        0.0f64.to_bits(),
        1,
        1.0f64.to_bits(),
        f64::INFINITY.to_bits(),
    ];
    let mut st = KernelState::default();
    st.sgpr[8] = 1 << 6 | 1 << 7 | 1 << 8 | 1 << 9; // positive non-NaN classes
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let class = l % classes.len();
        let bits = classes[class];
        (r[50], r[51]) = (bits as u32, (bits >> 32) as u32);
        r[52] = 1 << class;
        r[53] = 1 << 6;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, u32::MAX), "each lane matches its f64 class (vcc)");
    assert_eq!((r.sgpr[20], r.sgpr[21]), (u32::MAX, u32::MAX), "each lane matches its f64 class (sdst)");
    // |x| ∈ {+0, +denormal, +normal, +inf} exactly when x is not a NaN: negative lanes fold
    let pos: u64 = (0..64).filter(|l| l % 10 >= 2).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[22] as u64 | (r.sgpr[23] as u64) << 32, pos, "abs: every non-NaN becomes positive");
    let zero: u64 = (0..64).filter(|l| l % 10 == 6).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[126] as u64 | (r.sgpr[127] as u64) << 32, zero, "cmpx keeps positive-zero lanes");
}

/// Normalized f16 conversions – v_cvt_norm_u16_f16 / _i16_f16 (VOP1 single half-word
/// result, VOP3 with op_sel picking source/destination halves) and v_cvt_pknorm_u16_f16 /
/// _i16_f16 (VOP3, two f16 halves → a packed dword). Emitted text is deterministic.
#[test]
fn cvt_norm_f16_instructions() {
    let code = prog(&[
        ("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 9)], None),
        ("V_CVT_NORM_I16_F16", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 9), ("SRC1", V + 10), ("SRC2", V + 11), ("OP_SEL", 0b1001)], None),
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V + 9), ("SRC1", V + 10)], None),
        ("V_CVT_PKNORM_I16_F16", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 9), ("SRC1", V + 10), ("OP_SEL", 0b0011)], None),
        ("V_CVT_PKNORM_U16_F16", "ENC_VOP3", &[("VDST", 5), ("SRC0", 242), ("SRC1", 243)], None), // f16 +1.0, −1.0
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    // single f16 source (low half) → unsigned / signed normalized short, 16-bit result
    assert!(t.hlsl.contains("round(saturate(f16tof32(((v9) & 0xFFFFu))) * 65535.0f)"), "{}", t.hlsl);
    // op_sel[0] reads the source's high half, op_sel[3] writes the destination's high
    // half preserving the low one; abs/neg apply on the f16 value
    assert!(t.hlsl.contains("round(clamp(f16tof32(((v9) >> 16)), -1.0f, 1.0f) * 32767.0f)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(v2 & 0xFFFFu) | (t") && t.hlsl.contains("<< 16), v2)"), "{}", t.hlsl);
    // packed: two whole dwords written; each source half picked by op_sel
    assert!(t.hlsl.contains("round(saturate(f16tof32(v9 & 0xFFFFu)) * 65535.0f) | ((uint)round(saturate(f16tof32(v10 & 0xFFFFu)) * 65535.0f) << 16"), "{}", t.hlsl);
    assert!(t.hlsl.contains("round(clamp(f16tof32(v9 >> 16), -1.0f, 1.0f) * 32767.0f) & 0xFFFFu) | (((uint)(int)round(clamp(f16tof32(v10 >> 16), -1.0f, 1.0f) * 32767.0f) & 0xFFFFu) << 16"), "{}", t.hlsl);
    // inline float constants decode as f16 bit patterns (1.0 = 0x3C00, −1.0 = 0xBC00)
    assert!(t.hlsl.contains("f16tof32(0x00003C00u & 0xFFFFu)") && t.hlsl.contains("f16tof32(0x0000BC00u & 0xFFFFu)"), "{}", t.hlsl);
}

/// The f64 frexp pair: v_frexp_mant_f64 (m in [0.5,1) keeping the sign, input back for
/// 0/inf/NaN) → an f64 pair, v_frexp_exp_i32_f64 (input = m·2^exp, 0 for 0/inf/NaN) → one
/// dword. VOP1 and VOP3 (source abs/neg come from the f64 modifiers). Emitted text is
/// deterministic; helpers normalise denormals bit-exactly.
#[test]
fn frexp_f64_instructions() {
    let code = prog(&[
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 10)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 10), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 10), ("ABS", 1)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 8), ("SRC0", 243)], None), // inline constant −1.0
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // the source pair becomes a double; the mantissa stays f64, the exponent is one dword
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("(uint)frexp_exp_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("frexp_exp_d(asdouble(v10, (v11 & 0x7FFFFFFFu)))"), "{}", t.hlsl);
    // the f64 pair of inline −1.0 = 0xBFF0000000000000
    assert!(t.hlsl.contains("frexp_mant_d(asdouble(0x00000000u, 0xBFF00000u))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn rndne_f64_instructions() {
    let code = prog(&[
        ("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 10)], None),
        ("V_RNDNE_F64", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 10), ("NEG", 1)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("round_ne_d(asdouble(v10, v11))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("round_ne_d(asdouble(v10, (v11 ^ 0x80000000u)))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_rndne_f64() {
    if !gpu_ready_wave32() { return; }
    let p = prog(&[("V_RNDNE_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None)]);
    const INPUTS: [u64; 16] = [
        0.0f64.to_bits(), (-0.0f64).to_bits(), 0.5f64.to_bits(), (-0.5f64).to_bits(),
        1.5f64.to_bits(), 2.5f64.to_bits(), (-1.5f64).to_bits(), (-2.5f64).to_bits(),
        0.49999999999999994f64.to_bits(), 0.5000000000000001f64.to_bits(),
        4503599627370495.5f64.to_bits(), 4503599627370496.0f64.to_bits(),
        f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits(), 0x7FF8_0000_0000_1234, 1,
    ];
    let mut st = KernelState::default();
    for (lane, regs) in st.vgpr.iter_mut().enumerate() {
        let x = INPUTS[lane % INPUTS.len()];
        (regs[0], regs[1]) = (x as u32, (x >> 32) as u32);
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for (lane, regs) in r.vgpr.iter().enumerate() {
        let x = f64::from_bits(INPUTS[lane % INPUTS.len()]);
        let got = regs[2] as u64 | ((regs[3] as u64) << 32);
        let want = if x.is_nan() { x.to_bits() } else { x.round_ties_even().to_bits() };
        assert_eq!(got, want, "lane {lane}: {x:?}");
    }
}

#[test]
fn gpu_frexp_f64_exp_and_mant() {
    if !gpu_ready_wave32() {
        return;
    }
    let p = prog(&[
        ("V_FREXP_MANT_F64", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 0)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 0)], None),
        ("V_FREXP_MANT_F64", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 0), ("NEG", 1)], None),
        ("V_FREXP_EXP_I32_F64", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 0), ("ABS", 1)], None),
    ]);
    const INPUTS: [u64; 16] = [
        12.5f64.to_bits(),            // 0.78125 · 2^4
        (-4.0f64).to_bits(),          // −0.5 · 2^3
        0.5f64.to_bits(),             // 0.5 · 2^0
        1.0f64.to_bits(),             // 0.5 · 2^1
        0.0f64.to_bits(),             // special → 0/0
        (-0.0f64).to_bits(),          // special → input/0
        f64::INFINITY.to_bits(),      // special → input/0
        f64::NEG_INFINITY.to_bits(),  // special → input/0
        0x7FF8_0000_0000_0001,        // NaN → input payload, 0
        0x0000_0000_0000_0001,        // smallest denormal → 0.5 · 2^−1073
        0x0000_0000_000C_0000,        // denormal 3·2^−1056 → 0.75 · 2^−1054
        0x000F_FFFF_FFFF_FFFF,        // largest denormal → (1−2^−52) · 2^−1022
        0x0010_0000_0000_0000,        // smallest normal → 0.5 · 2^−1021
        0x7FEF_FFFF_FFFF_FFFF,        // max normal → (1−2^−53) · 2^1024
        0x4059_0000_0000_0000,        // 100.0 → 0.78125 · 2^7
        0x800F_FFFF_FFFF_FFFF,        // negative denormal → −(1−2^−52) · 2^−1022
    ];
    // CPU mirror of the ISA semantics: e = topmost set bit of the 52-bit fraction − 1073
    // for denormals, biased exponent − 1022 for normals; mant = normalised [0.5,1) value.
    fn frexp_ref(bits: u64) -> (u64, i32) {
        let (h, l) = ((bits >> 32) as u32, bits as u32);
        let (e, m, s) = ((h >> 20) & 0x7FF, h & 0xF_FFFF, h & 0x8000_0000);
        if e == 0x7FF || (e == 0 && (m | l) == 0) {
            return (bits, 0);
        }
        if e != 0 {
            return ((((s | m | 0x3FE0_0000) as u64) << 32) | l as u64, e as i32 - 1022);
        }
        let top = if m != 0 { 63 - m.leading_zeros() } else { 31 - l.leading_zeros() };
        let f = ((m as u64) << 32) | l as u64;
        let ff = (f << (52 - top)) & 0x000F_FFFF_FFFF_FFFF;
        ((((s | 0x3FE0_0000 | (ff >> 32) as u32) as u64) << 32) | (ff & 0xFFFF_FFFF), top as i32 - 1073)
    }
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let x = INPUTS[l % INPUTS.len()];
        r[0] = x as u32;
        r[1] = (x >> 32) as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (mant, exp) = frexp_ref(INPUTS[l % INPUTS.len()]);
        assert_eq!(v[2] as u64 | (v[3] as u64) << 32, mant, "lane {l} v_frexp_mant_f64");
        assert_eq!(v[4] as i32, exp, "lane {l} v_frexp_exp_i32_f64");
        // negated source: mantissa flips its sign, exponent is unchanged
        assert_eq!(v[6] as u64 | (v[7] as u64) << 32, mant ^ 0x8000_0000_0000_0000, "lane {l} v_frexp_mant_f64 neg");
        assert_eq!(v[8] as i32, exp, "lane {l} v_frexp_exp_i32_f64 abs");
    }
}

#[test]
fn gpu_valu_float_and_int() {
    if !gpu_ready_wave32() {
        return;
    }
    let code = prog(&[
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 10), ("SRC0", V + 1), ("VSRC1", 2)], None),
        ("V_MUL_LEGACY_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 128), ("VSRC1", 3)], None), // 0 * inf = 0
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", 244), ("SRC2", 243)], None), // v1*2-1
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 13), ("SRC0", V + 1), ("VSRC1", 2)], None), // v13 += v1*v2
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 14), ("SRC0", V + 4), ("VSRC1", 5)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 15), ("SRC0", 132), ("VSRC1", 4)], None), // v4 << 4
        ("V_BFE_U32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 6), ("SRC1", 136), ("SRC2", 132)], None), // bits 8..11
        ("V_CVT_F32_UBYTE1", "ENC_VOP1", &[("VDST", 17), ("SRC0", V + 6)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", 193), ("SRC1", 128)], None), // lane index (low half)
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 18), ("SRC0", 193), ("SRC1", V + 18)], None), // + high half
        ("V_MED3_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 1), ("SRC1", 128), ("SRC2", 242)], None), // clamp to [0,1]
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 1), ("SRC1", V + 2), ("NEG", 1), ("OMOD", 1)], None), // (-v1 + v2) * 2
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = f(l as f32 * 0.25);
        r[2] = f(1.5);
        r[3] = f(f32::INFINITY);
        r[4] = l as u32;
        r[5] = 1000;
        r[6] = 0x0000_AB00 | l as u32;
        r[13] = f(10.0);
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let x = l as f32 * 0.25;
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!(fl(10), x + 1.5, "lane {l} v_add_f32");
        assert_eq!(v[11], 0, "lane {l} v_mul_legacy_f32 0*inf");
        assert_eq!(fl(12), x * 2.0 - 1.0, "lane {l} v_fma_f32");
        assert_eq!(fl(13), 10.0 + x * 1.5, "lane {l} v_mac_f32");
        assert_eq!(v[14], l as u32 + 1000, "lane {l} v_add_nc_u32");
        assert_eq!(v[15], (l as u32) << 4, "lane {l} v_lshlrev_b32");
        assert_eq!(v[16], 0xB, "lane {l} v_bfe_u32");
        assert_eq!(fl(17), 0xAB as f32, "lane {l} v_cvt_f32_ubyte1");
        assert_eq!(v[18], l as u32, "lane {l} v_mbcnt");
        assert_eq!(fl(19), x.clamp(0.0, 1.0), "lane {l} v_med3_f32");
        assert_eq!(fl(20), (1.5 - x) * 2.0, "lane {l} VOP3 neg+omod");
    }
}

#[test]
fn gpu_bench_really_runs() {
    if !gpu_ready_wave32() {
        return;
    }
    println!("GPU: {} / DXC: {}", runner::adapter().unwrap(), super::dxc::location().unwrap().display());
    // v1 = lane * 3 must differ per lane – proves the kernel ran on the GPU, per lane
    let code = prog(&[("V_MUL_U32_U24", "ENC_VOP2", &[("VDST", 1), ("SRC0", 131), ("VSRC1", 0)], None)]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[1] = 0xDEAD;
    }
    // EXEC = only even lanes
    st.sgpr[126] = 0x5555_5555;
    st.sgpr[127] = 0x5555_5555;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        let want = if l % 2 == 0 { 3 * l as u32 } else { 0xDEAD };
        assert_eq!(r.vgpr[l][1], want, "lane {l}");
    }
    assert_eq!((r.sgpr[126], r.sgpr[127]), (0x5555_5555, 0x5555_5555));
}

/// A wave64 kernel as two native wave32 halves (an adapter without 64-lane waves): each half
/// writes its own word of a lane mask, and the result has both; uniform SGPRs stay as they are.
#[test]
fn gpu_kernel_halves_merge_lane_masks() {
    if !gpu_ready_wave32() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)), // vcc = v0 < 40
        ("S_MOVK_I32", "ENC_SOPK", &[("SDST", 5), ("SIMM16", 0x1234)], None),
    ]);
    let t = translate(&code, &Options { hw_wave: 32, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(32)]") && t.hlsl.contains("[numthreads(64, 1, 1)]"), "{}", t.hlsl);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let mut st = KernelState::default();
    st.sgpr[126] = u32::MAX;
    st.sgpr[127] = u32::MAX;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
    }
    let r = runner::run(&dxil, &st, &[]).unwrap();
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, 0xFF), "vcc");
    assert_eq!(r.sgpr[5], 0x1234, "uniform s5");
    assert_eq!((r.sgpr[126], r.sgpr[127]), (u32::MAX, u32::MAX), "exec");
}

#[test]
fn gpu_scalar_loop_and_scc() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 128)], None),             // 0: s0 = 0
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 138)], None),             // 4: s1 = 10
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 0), ("SSRC1", 1)], None),  // 8: loop: s0 += s1
        ("S_SUB_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)], None), // 12: s1 -= 1
        ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 128)], None),         // 16
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)], None),                 // 20: → 8
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),                // v1 = s0
        // SCC rules
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 255), ("SSRC1", 130)], Some(0xFFFF_FFFF)), // carry
        ("S_ADDC_U32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 128), ("SSRC1", 128)], None),             // 0+0+carry
        ("S_ADD_I32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 255), ("SSRC1", 129)], Some(0x7FFF_FFFF)), // overflow
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 135), ("SSRC1", 136)], None),           // scc ? 7 : 8
        ("S_BFE_U32", "ENC_SOP2", &[("SDST", 6), ("SSRC0", 255), ("SSRC1", 20)], Some(0x00AB_CD00)), // s20 = (8 << 16) | 8
        ("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 255)], Some(0x0000_0100)),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 8), ("SSRC0", 126)], None), // popcount(exec)
        ("S_LSHL_B64", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 255), ("SSRC1", 160)], Some(0x0000_0003)), // 3 << 32
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 193), ("SSRC1", 255)], Some(1234)),  // -1 * 1234
        ("S_BFE_U64", "ENC_SOP2", &[("SDST", 14), ("SSRC0", 24), ("SSRC1", 22)], None),
        ("S_BFE_I64", "ENC_SOP2", &[("SDST", 16), ("SSRC0", 24), ("SSRC1", 23)], None),
        ("S_BITCMP1_B64", "ENC_SOPC", &[("SSRC0", 24), ("SSRC1", 26)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 27), ("SSRC0", 135), ("SSRC1", 136)], None),
        ("S_BITCMP0_B64", "ENC_SOPC", &[("SSRC0", 24), ("SSRC1", 28)], None),
        ("S_CSELECT_B32", "ENC_SOP2", &[("SDST", 29), ("SSRC0", 135), ("SSRC1", 136)], None),
        ("S_BCNT0_I32_B64", "ENC_SOP1", &[("SDST", 34), ("SSRC0", 24)], None),
        ("S_FF0_I32_B64", "ENC_SOP1", &[("SDST", 35), ("SSRC0", 24)], None),
        ("S_FLBIT_I32_B64", "ENC_SOP1", &[("SDST", 36), ("SSRC0", 24)], None),
        ("S_BITSET1_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 32)], None),
        ("S_BITSET0_B64", "ENC_SOP1", &[("SDST", 30), ("SSRC0", 33)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 37), ("SSRC0", 24)], None),
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 38), ("SSRC0", 24)], None),
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 39), ("SSRC0", 128)], None),       // 0: nothing differs from the sign
        ("S_FLBIT_I32", "ENC_SOP1", &[("SDST", 40), ("SSRC0", 193)], None),       // -1: all bits are the sign
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 41), ("SSRC0", 44)], None),    // high half 0 → scan the low half
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 42), ("SSRC0", 46)], None),    // negative, ~high = 0 → scan the low half
        ("S_FLBIT_I32_I64", "ENC_SOP1", &[("SDST", 43), ("SSRC0", 128)], None),   // 0 → all bits equal the sign
    ]);
    let mut st = KernelState::default();
    st.sgpr[20] = (8 << 16) | 8;
    st.sgpr[22] = (20 << 16) | 12;
    st.sgpr[23] = (4 << 16) | 60;
    st.sgpr[24] = 0x7654_3210;
    st.sgpr[25] = 0xFEDC_BA98;
    st.sgpr[26] = 63;
    st.sgpr[28] = 0;
    st.sgpr[30] = 1 << 4;
    st.sgpr[31] = 0;
    st.sgpr[32] = 36;
    st.sgpr[33] = 4;
    st.sgpr[44] = 0x8000;                                                        // s[44:45] = 0x0000_0000_0000_8000
    st.sgpr[45] = 0;
    st.sgpr[46] = 0x8000_0000;                                                   // s[46:47] = 0xFFFF_FFFF_8000_0000
    st.sgpr[47] = 0xFFFF_FFFF;
    let r = runner::run_code(&code, &st, &[]).unwrap();
    assert_eq!(r.sgpr[0], 55, "loop");
    assert!(r.vgpr.iter().all(|v| v[1] == 55));
    assert_eq!(r.sgpr[2], 1, "s_add_u32 wraps");
    assert_eq!(r.sgpr[3], 1, "s_addc_u32 uses the carry");
    assert_eq!(r.sgpr[4], 0x8000_0000, "s_add_i32");
    assert_eq!(r.sgpr[5], 7, "s_cselect after signed overflow → scc=1");
    assert_eq!(r.sgpr[6], 0xCD, "s_bfe_u32");
    assert_eq!(r.sgpr[7], 8, "s_ff1");
    assert_eq!(r.sgpr[8], 64, "s_bcnt1 exec");
    assert_eq!((r.sgpr[10], r.sgpr[11]), (0, 3), "s_lshl_b64");
    assert_eq!(r.sgpr[12] as i32, -1234, "s_mul_i32");
    assert_eq!((r.sgpr[14], r.sgpr[15]), (0x0007_6543, 0), "s_bfe_u64");
    assert_eq!((r.sgpr[16], r.sgpr[17]), (u32::MAX, u32::MAX), "s_bfe_i64 sign extension");
    assert_eq!((r.sgpr[27], r.sgpr[29]), (7, 7), "s_bitcmp1/0_b64");
    assert_eq!((r.sgpr[34], r.sgpr[35], r.sgpr[36]), (32, 0, 0), "s_bcnt0/ff0/flbit 64");
    assert_eq!((r.sgpr[30], r.sgpr[31]), (0, 1 << 4), "s_bitset1/0_b64");
    assert_eq!((r.sgpr[37], r.sgpr[38]), (1, 7), "signed scalar leading-bit scans");
    assert_eq!((r.sgpr[39], r.sgpr[40]), (u32::MAX, u32::MAX), "flbit: nothing differs from the sign");
    assert_eq!((r.sgpr[41], r.sgpr[42], r.sgpr[43]), (48, 33, u32::MAX), "flbit_i64 high-then-low scan");
}

#[test]
fn gpu_divergence_through_exec() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 160), ("VSRC1", 0)], None), // vcc = 32 > lane
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 129)], None), // lanes 0..31: 1
        ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 130)], None), // lanes 32..63: 2
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 2)], None),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 128), ("VSRC1", 3)], None), // vcc ? v3 : 0
        ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V), ("SRC1", 255)], Some(45)),
        // nobody matches: exec = 0 → the block is skipped
        ("V_CMPX_EQ_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(1000)), // 36: (literal: 8 bytes)
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None),                    // 44 → 56
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 192)], None), // 48 (skipped)
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 129)], None),             // 52 (skipped)
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)], None),           // 56: exec = -1
        ("V_ADD_CO_U32", "VOP3_SDST_ENC", &[("VDST", 8), ("SDST", 20), ("SRC0", V + 9), ("SRC1", 255)], Some(0x8000_0000)),
    ]);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[3] = 7 + l as u32;
        r[6] = 0x66;
        r[9] = if l % 4 == 0 { 0x8000_0000 } else { 1 };
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!(r.vgpr[l][1], if l < 32 { 1 } else { 2 }, "lane {l} masked writes");
        assert_eq!(r.vgpr[l][2], if l < 32 { 7 + l as u32 } else { 0 }, "lane {l} cndmask");
        assert_eq!(r.vgpr[l][6], 0x66, "lane {l}: skipped block must not run");
    }
    assert_eq!((r.sgpr[2], r.sgpr[3]), (u32::MAX, u32::MAX), "saved exec");
    assert_eq!((r.sgpr[106], r.sgpr[107]), (u32::MAX, 0), "vcc from v_cmp");
    assert_eq!(r.sgpr[4], 7, "readfirstlane");
    assert_eq!(r.sgpr[5], 45, "readlane 45");
    assert_eq!(r.sgpr[7], 0, "scalar code in the skipped block");
    assert_eq!((r.sgpr[126], r.sgpr[127]), (u32::MAX, u32::MAX));
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0x1111_1111, 0x1111_1111), "carry mask of v_add_co_u32");
}

#[test]
fn gpu_memory_and_exports() {
    if !gpu_ready_wave32() {
        return;
    }
    let code = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 6), ("OFFSET", 4), ("SOFFSET", 125)], None),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 11), ("SBASE", 6), ("OFFSET", 0x400), ("SOFFSET", 125)], None), // out of range
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("OFFEN", 1)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 8)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 3), ("VSRC1", 4), ("VSRC2", 0), ("VSRC3", 0), ("DONE", 1)], None),
        ("V_CVT_PKRTZ_F16_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 242), ("VSRC1", 6)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0x3), ("VSRC0", 5), ("VSRC1", 5), ("COMPR", 1), ("DONE", 1), ("VM", 1)], None),
    ]);
    let mut mem = vec![0u32; 256];
    for (i, m) in mem.iter_mut().enumerate() {
        *m = 0x1000 + i as u32;
    }
    let mut st = KernelState::default();
    st.sgpr[0] = 0; // flat pointer
    st.sgpr[12] = 0x40; // V#: base 0x40, stride 0, 0x100 bytes
    st.sgpr[13] = 0;
    st.sgpr[14] = 0x100;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 4 * l as u32;
        r[6] = f(0.5);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!((r.sgpr[8], r.sgpr[9]), (0x1004, 0x1005), "s_load_dwordx2");
    assert_eq!(r.sgpr[10], 0x1000 + 0x44 / 4, "s_buffer_load_dword");
    assert_eq!(r.sgpr[11], 0, "s_buffer_load out of range → 0");
    for l in 0..64 {
        let want = if 8 + 4 * l < 0x100 { 0x1000 + (0x40 + 8 + 4 * l as u32) / 4 } else { 0 };
        assert_eq!(r.vgpr[l][3], want, "lane {l} buffer_load_dword offen");
        assert_eq!(r.export(12, l, 0), want, "lane {l} pos0.x");
        assert_eq!(r.export(12, l, 1), 0x1004, "lane {l} pos0.y");
        assert_eq!(r.export(12, l, 2), 4 * l as u32, "lane {l} pos0.z");
        assert_eq!((r.export(0, l, 0), r.export(0, l, 1)), (0x3C00, 0x3800), "lane {l} mrt0 compr (1.0, 0.5)");
    }
}

/// A real console vertex shader from the retail pack, run on the PC GPU
/// (needs XCL_SAMPLE_RPF=…\shaders.rpf).
#[test]
fn gpu_retail_vs_blit_gen_amb_cube() {
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else {
        println!("XCL_SAMPLE_RPF not set – skipping");
        return;
    };
    if !gpu_ready() {
        return;
    }
    let data = std::fs::read(&path).unwrap();
    let rec = crate::gpu::pack::scan(&data, 0x40).into_iter().find(|r| r.name == "VS_BlitGenAmbCube_0cfdb358_Wrapped").expect("record");
    let blob = crate::gpu::pack::inflate(&data, &rec).unwrap();
    let sh = crate::gpu::xbox::parse(&blob).unwrap();
    let fs = &sh.fssm[0];
    let p = fs.programs[0];
    let code = &fs.data[p.offset..p.offset + p.len];
    let t = translate(code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported, t.hlsl);
    println!("{}", t.hlsl);
    // the fetch shader would load POSITION into v4..; give every lane its own vertex
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[4] = f((l % 8) as f32 / 7.0);
        r[5] = f((l / 8) as f32 / 7.0);
    }
    let (r, _) = runner::run_program(code, &st, &[]).unwrap();
    for l in 0..64 {
        let x = (l % 8) as f32 / 7.0;
        let y = (l / 8) as f32 / 7.0;
        let pos: Vec<f32> = (0..4).map(|c| f32::from_bits(r.export(12, l, c))).collect();
        assert_eq!(pos, vec![x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0], "lane {l}: SV_Position");
        let p0: Vec<f32> = (0..4).map(|c| f32::from_bits(r.export(32, l, c))).collect();
        assert_eq!(p0, vec![0.0, 0.0, 0.0, 1.0], "lane {l}: param0");
    }
}

#[test]
fn gpu_sdwa_dpp_and_cross_lane() {
    if !gpu_ready() {
        return;
    }
    let code = prog(&[
        // v10 = (float)(v1 byte 2)      SRC0=0xF9 (SDWA), VSRC0=1, SRC0_SEL=BYTE_2, DST_SEL=DWORD
        ("V_CVT_F32_U32", "VOP1_VOP_SDWA", &[("VDST", 10), ("SRC0", 0xF9), ("VSRC0", 1), ("SRC0_SEL", 2), ("DST_SEL", 6)], None),
        // v11 = sext(v1 word1) + v2 written into word 0 of v11, upper word preserved
        ("V_ADD_NC_U32", "VOP2_VOP_SDWA", &[("VDST", 11), ("SRC0", 0xF9), ("VSRC0", 1), ("VSRC1", 2), ("SRC0_SEL", 5), ("SRC0_SEXT", 1), ("SRC1_SEL", 6), ("DST_SEL", 4), ("DST_UNUSED", 2)], None),
        // v12 = v0 of lane (lane ^ 1 within quad): quad_perm [1,0,3,2] = 0b10_11_00_01 = 0xB1
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 12), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0xB1), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
        // v13 = v0 of lane-1 within row; lane 0 of each row has no source → 0 (bound_ctrl)
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 13), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)], None),
        // v14 = row_mirror, rows 1 and 3 only (row_mask 0b1010)
        ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 14), ("SRC0", 0xFA), ("VSRC0", 0), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xA), ("BANK_MASK", 0xF)], None),
        // v15 = v0 of lane 7 - (lane & 7) in each group of 8 (DPP8)
        ("V_MOV_B32", "VOP1_VOP_DPP8", &[("VDST", 15), ("SRC0", 0xE9), ("VSRC0", 0), ("LANE_SEL_0", 7), ("LANE_SEL_1", 6), ("LANE_SEL_2", 5), ("LANE_SEL_3", 4), ("LANE_SEL_4", 3), ("LANE_SEL_5", 2), ("LANE_SEL_6", 1), ("LANE_SEL_7", 0)], None),
        // v16 = swizzle: lane ^ 4 (BitMode: and 0x1F, or 0, xor 4)
        ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 16), ("ADDR", 0), ("OFFSET0", 0x1F), ("OFFSET1", 4 << 2)], None),
        // v17 = permlanex16 with identity selects → lane ^ 16
        ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V), ("SRC1", 255), ("SRC2", 3)], Some(0x7654_3210)),
        // v18 = fma_mix(f16 lo of v3, f16 hi of v3, f32 v4) = 1.5 * 2.0 + 0.25
        ("V_FMA_MIX_F32", "ENC_VOP3P", &[("VDST", 18), ("SRC0", V + 3), ("SRC1", V + 3), ("SRC2", V + 4), ("OP_SEL", 0b010), ("OP_SEL_HI", 0b11), ("OP_SEL_HI_2", 0)], None),
        // v19 = cube face of (v5, v6, v7)
        ("V_CUBEID_F32", "ENC_VOP3", &[("VDST", 19), ("SRC0", V + 5), ("SRC1", V + 6), ("SRC2", V + 7)], None),
    ]);
    let mut st = KernelState::default();
    st.sgpr[3] = 0xFEDC_BA98; // permlanex16 selects for lanes 8..15 of a row
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 100 + l as u32;
        r[1] = 0x8001_0000 | ((l as u32) << 16 & 0x00F0_0000) ;
        r[2] = 10;
        r[11] = 0xAAAA_5555;
        r[14] = 0xDEAD;
        r[3] = 0x4000_3E00; // hi = 2.0h, lo = 1.5h
        r[4] = f(0.25);
        let axis = l % 3;
        r[5] = f(if axis == 0 { -3.0 } else { 0.5 });
        r[6] = f(if axis == 1 { 3.0 } else { 0.5 });
        r[7] = f(if axis == 2 { -3.0 } else { 0.5 });
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let v1 = 0x8001_0000u32 | ((l as u32) << 16 & 0x00F0_0000);
        assert_eq!(f32::from_bits(v[10]), ((v1 >> 16) & 0xFF) as f32, "lane {l} sdwa byte2");
        let w1 = ((v1 >> 16) as u16 as i16 as i32) as u32;
        assert_eq!(v[11], 0xAAAA_0000 | (w1.wrapping_add(10) & 0xFFFF), "lane {l} sdwa word1 sext + preserve");
        assert_eq!(v[12], 100 + (l ^ 1) as u32, "lane {l} quad_perm");
        assert_eq!(v[13], if l % 16 == 0 { 0 } else { 100 + l as u32 - 1 }, "lane {l} row_shr:1 bound_ctrl");
        let row = l / 16;
        assert_eq!(v[14], if row % 2 == 1 { 100 + (row * 16 + 15 - l % 16) as u32 } else { 0xDEAD }, "lane {l} row_mirror row_mask");
        assert_eq!(v[15], 100 + ((l & !7) | (7 - (l & 7))) as u32, "lane {l} dpp8");
        assert_eq!(v[16], 100 + (l ^ 4) as u32, "lane {l} ds_swizzle");
        let n = if l % 16 < 8 { (0x7654_3210u32 >> (4 * (l % 8))) & 15 } else { (0xFEDC_BA98u32 >> (4 * (l % 8))) & 15 };
        assert_eq!(v[17], 100 + ((l & !31) | ((l & 16) ^ 16) | n as usize) as u32, "lane {l} permlanex16");
        assert_eq!(f32::from_bits(v[18]), 1.5 * 2.0 + 0.25, "lane {l} fma_mix");
        let face = match l % 3 {
            0 => 1.0,
            1 => 2.0,
            _ => 5.0,
        };
        assert_eq!(f32::from_bits(v[19]), face, "lane {l} cubeid");
    }
}

#[test]
fn buffer_format_table_matches_llvm() {
    // every BUF_FMT_* in LLVM's gfx10 MTBUF tests must decode to the same name
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/llvm-amdgpu");
    let mut checked = 0;
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        for line in std::fs::read_to_string(e.path()).unwrap_or_default().lines() {
            let Some(i) = line.find("format:[BUF_FMT_") else { continue };
            if !line.starts_with("# GFX10") {
                continue;
            }
            let name = &line[i + 16..line[i..].find(']').unwrap() + i];
            let Some(enc) = line.split("encoding: [").nth(1) else { continue };
            let b: Vec<u8> = enc.trim_end_matches(']').split(',').filter_map(|x| u8::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok()).collect();
            let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            assert_eq!(super::emit::BUF_FORMATS[((w >> 19) & 0x7F) as usize], name, "{line}");
            checked += 1;
        }
    }
    assert!(checked >= 10, "{checked}");
}

#[test]
fn buffer_format_d16_packs_and_unpacks_halves() {
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_D16_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 12), ("SRSRC", 3), ("SOFFSET", 128)], None),
        ("TBUFFER_STORE_FORMAT_D16_XYZ", "ENC_MTBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128), ("FORMAT", 71)], None),
        ("BUFFER_STORE_FORMAT_D16_HI_X", "ENC_MUBUF", &[("VDATA", 16), ("SRSRC", 3), ("SOFFSET", 128)], None),
    ]);
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    assert!(t.hlsl.contains("v10 = t1_ok ? (select(t1_int") && t.hlsl.contains("v11 = t1_ok ? (select(t1_int"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v12 = (v12 & 0xFFFFu) | (t2_ok ? (select(t2_int") && t.hlsl.contains("<< 16) : 0u)"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint t3_ty = (t3_info >> 3) & 7u") && t.hlsl.contains("f16tof32(((v14) & 0xFFFFu))"), "{}", t.hlsl);
    assert!(t.hlsl.contains("f16tof32(((v16 >> 16) & 0xFFFFu))"), "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

#[test]
fn gpu_typed_buffer_loads_and_stores() {
    if !gpu_ready_wave32() {
        return;
    }
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let bgra = 6 | 5 << 3 | 4 << 6 | 7 << 9;
    let vs = |base: u32, fmt: u32, sel: u32| [base, 0, 0x100, sel | fmt << 12];
    let code = prog(&[
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 10), ("SRSRC", 3), ("SOFFSET", 128)], None),             // A
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 14), ("SRSRC", 4), ("SOFFSET", 128)], None),             // B
        ("BUFFER_LOAD_FORMAT_XYZW", "ENC_MUBUF", &[("VDATA", 18), ("SRSRC", 5), ("SOFFSET", 128)], None),             // C
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 22), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 2)], None), // D (unaligned)
        ("BUFFER_LOAD_FORMAT_XYZ", "ENC_MUBUF", &[("VDATA", 23), ("SRSRC", 7), ("SOFFSET", 128)], None),              // E
        ("TBUFFER_LOAD_FORMAT_XY", "ENC_MTBUF", &[("VDATA", 26), ("SRSRC", 8), ("SOFFSET", 128), ("FORMAT", 64)], None), // F
        ("BUFFER_LOAD_FORMAT_X", "ENC_MUBUF", &[("VDATA", 28), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 0x200)], None), // out of range
        ("BUFFER_LOAD_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 31), ("SRSRC", 4), ("SOFFSET", 128)], None),
        ("BUFFER_LOAD_FORMAT_D16_X", "ENC_MUBUF", &[("VDATA", 32), ("SRSRC", 6), ("SOFFSET", 128), ("OFFSET", 2)], None),
        ("TBUFFER_LOAD_FORMAT_D16_XY", "ENC_MTBUF", &[("VDATA", 33), ("SRSRC", 8), ("SOFFSET", 128), ("FORMAT", 64)], None),
        ("BUFFER_STORE_FORMAT_D16_XY", "ENC_MUBUF", &[("VDATA", 34), ("SRSRC", 10), ("SOFFSET", 128)], None),
        ("V_CVT_PK_U8_F32", "ENC_VOP3", &[("VDST", 29), ("SRC0", V + 30), ("SRC1", 130), ("SRC2", 255)], Some(0x1122_3344)),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 0), ("VADDR", 1), ("SRSRC", 9), ("SOFFSET", 128), ("OFFEN", 1)], None),
        ("S_BITSET1_B32", "ENC_SOP1", &[("SDST", 50), ("SSRC0", 133)], None),
    ]);
    let mut mem = vec![0u32; 512];
    mem[0x100 / 4] = u32::from_le_bytes([0x00, 0x80, 0xFF, 0x40]); // A: 8_8_8_8_UNORM
    mem[0x110 / 4] = 0x3C00 | 0xC000 << 16; // B: 16_16_FLOAT (1.0, -2.0)
    mem[0x114 / 4] = 1023 | 512 << 20 | 3 << 30; // C: 2_10_10_10_UNORM
    mem[0x118 / 4] = 0x8001 << 16; // D: 16_SNORM at byte 0x11A
    mem[0x11C / 4] = 0x3C0 | 0x400 << 11 | 0x1C0 << 22; // E: 10_11_11_FLOAT (1.0, 2.0, 0.5)
    mem[0x120 / 4] = f(1.5);
    mem[0x124 / 4] = f(-3.25); // F: 32_32_FLOAT
    let mut st = KernelState::default();
    for (i, v) in [
        vs(0x100, 56, bgra),
        vs(0x110, 29, ident),
        vs(0x114, 50, ident),
        vs(0x118, 8, ident),
        vs(0x11C, 36, ident),
        vs(0x120, 0, ident),
        [0x400, 0, 0x100, 0],
    ]
    .iter()
    .enumerate()
    {
        st.sgpr[12 + 4 * i..16 + 4 * i].copy_from_slice(v);
    }
    st.sgpr[40..44].copy_from_slice(&vs(0x130, 29, ident));
    st.sgpr[50] = 1;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 0xC0DE_0000 | l as u32;
        r[1] = 4 * l as u32;
        r[30] = f(200.6);
        r[34] = 0xB800_4000; // packed f16: +2.0, -0.5
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let fl = |x: u32| f32::from_bits(x);
    // normalised mid values: within 2 ULP (DXC divides approximately); endpoints exact
    let near = |got: u32, want: f32| (got as i64 - want.to_bits() as i64).abs() <= 2;
    for l in 0..64 {
        let v = &r.vgpr[l];
        assert!(
            v[10] == f(1.0) && near(v[11], 128.0 / 255.0) && v[12] == 0 && near(v[13], 64.0 / 255.0),
            "lane {l} A unorm8 bgra {:?}",
            [fl(v[10]), fl(v[11]), fl(v[12]), fl(v[13])]
        );
        assert_eq!([fl(v[14]), fl(v[15]), fl(v[16]), fl(v[17])], [1.0, -2.0, 0.0, 1.0], "lane {l} B 16_16_float");
        assert!(
            v[18] == f(1.0) && v[19] == 0 && near(v[20], 512.0 / 1023.0) && v[21] == f(1.0),
            "lane {l} C 2_10_10_10 {:?}",
            [fl(v[18]), fl(v[19]), fl(v[20]), fl(v[21])]
        );
        assert_eq!(fl(v[22]), -1.0, "lane {l} D 16_snorm unaligned");
        assert_eq!([fl(v[23]), fl(v[24]), fl(v[25])], [1.0, 2.0, 0.5], "lane {l} E 10_11_11_float");
        assert_eq!([fl(v[26]), fl(v[27])], [1.5, -3.25], "lane {l} F tbuffer 32_32_float");
        assert_eq!(v[28], 0, "lane {l} out of range");
        assert_eq!(v[31], 0xC000_3C00, "lane {l} d16 16_16_float");
        assert_eq!(v[32], 0x0000_BC00, "lane {l} d16 16_snorm");
        assert_eq!(v[33], 0xC280_3E00, "lane {l} d16 tbuffer 32_32_float");
        assert_eq!(v[29], 0x11C9_3344, "lane {l} v_cvt_pk_u8_f32 into byte 2");
        assert_eq!(r.mem[0x400 / 4 + l], 0xC0DE_0000 | l as u32, "lane {l} buffer_store_dword");
    }
    assert_eq!(r.mem[0x130 / 4], 0xB800_4000, "d16 store 16_16_float");
    assert_eq!(r.sgpr[50], 1 | 1 << 5, "s_bitset1");
}

fn el(name: &str, index: u32, sv: u32, ct: u32, reg: u32, mask: u8) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index, system_value: sv, component_type: ct, register: reg, mask, rw_mask: mask, stream: 0 }
}

/// VS + PS pair recompiled to graphics stages and drawn on the GPU: VS inputs from the
/// fetch-shader VGPRs, exp pos0/param0, PS interpolation through v_interp, discard via v_cmpx.
#[test]
fn gpu_graphics_vs_ps_draw() {
    if !gpu_ready() {
        return;
    }
    use super::{PsInterface, VsInterface};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32_FLOAT};
    let vs_code = prog(&[
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V + 4)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V + 5)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 242)], None),
        ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 0), ("VSRC1", 1), ("VSRC2", 2), ("VSRC3", 3), ("DONE", 1)], None),
        ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
    ]);
    let vsi = VsInterface {
        inputs: vec![el("POSITION", 0, 0, 3, 0, 3), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        attr_base: 4,
        params: None,
    };
    let ps_code = prog(&[
        ("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(8.0f32.to_bits())), // keep pos.x > 8
        ("V_INTERP_P1_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 0)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 5), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 1)], None),
        ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 6), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", 2)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)], None),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 240), ("VSRC1", 4)], None),
        ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1), ("VM", 1)], None),
    ]);
    let psi = PsInterface {
        inputs: vec![el("SV_Position", 0, 1, 3, 0, 0xF), el("COLOR", 0, 0, 3, 1, 0xF)],
        outputs: vec![el("SV_Target", 0, 64, 3, 0, 0xF)],
        input_ena: 0x302,
        attrs: None,
    };
    let vt = translate(&vs_code, &Options { mode: Mode::Vertex(vsi), wave64: true, ..Default::default() }).unwrap();
    let pt = translate(&ps_code, &Options { mode: Mode::Pixel(psi), wave64: true, ..Default::default() }).unwrap();
    assert!(vt.complete() && pt.complete(), "{:?} {:?}", vt.unsupported, pt.unsupported);
    let vs = super::dxc::compile(&vt.hlsl, "main", "vs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", vt.hlsl));
    let ps = super::dxc::compile(&pt.hlsl, "main", "ps_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", pt.hlsl));
    let c = [0.8f32, 0.6, 0.4, 1.0];
    let verts: Vec<f32> = [[-1.0f32, -1.0], [-1.0, 3.0], [3.0, -1.0]].iter().flat_map(|p| [p[0], p[1], c[0], c[1], c[2], c[3]]).collect();
    let layout = [
        runner::VertexElement { semantic: "POSITION", index: 0, format: DXGI_FORMAT_R32G32_FLOAT, offset: 0 },
        runner::VertexElement { semantic: "COLOR", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 8 },
    ];
    let clear = [0.0, 0.0, 1.0, 1.0];
    let px = runner::draw(&vs, &ps, &layout, &verts, 24, &[], &[], 16, clear).unwrap();
    for y in 0..16 {
        for x in 0..16 {
            let want = if x >= 8 { [0.4, 0.6, 0.4, 1.0] } else { clear };
            let got = px[y * 16 + x];
            assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5), "pixel ({x},{y}) = {got:?}, expected {want:?}");
        }
    }
}

/// XCL image descriptor (T#, GFX10 layout): BASE_ADDRESS = SRV heap index, FORMAT, TYPE.
fn tsharp(index: u32, format: u32, ty: u32) -> [u32; 8] {
    [index, format << 20, 0, ty << 28, 0, 0, 0, 0]
}

#[test]
fn gpu_images_sample_load_store_atomic() {
    if !gpu_ready_wave32() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32_UINT};
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // T# tex0 in s[16:23] (SRSRC = 4), S# point in s[24:27] (SSAMP = 6), S# linear in s[28:31] (7)
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)]));
    code.extend(mimg("IMAGE_SAMPLE_L", &[("VDATA", 20), ("VADDR", 30), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)]));
    code.extend(mimg("IMAGE_GATHER4_LZ", &[("VDATA", 40), ("VADDR", 33), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x1), ("DIM", 1)]));
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 14), ("VADDR", 3), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_GET_RESINFO", &[("VDATA", 50), ("VADDR", 5), ("SRSRC", 4), ("DMASK", 0xF), ("DIM", 1)]));
    // tex1: R32_UINT UAV, T# in s[32:39] (SRSRC = 8); tex2: 1x1 R32_UINT UAV in s[40:47] (10)
    code.extend(mimg("IMAGE_STORE", &[("VDATA", 6), ("VADDR", 37), ("SRSRC", 8), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(mimg("IMAGE_ATOMIC_UMAX", &[("VDATA", 7), ("VADDR", 35), ("SRSRC", 10), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    // tex3: 1x1 R32_UINT UAV in s[48:55] (12): image_atomic_inc (no HLSL intrinsic: CAS loop)
    code.extend(mimg("IMAGE_ATOMIC_INC", &[("VDATA", 39), ("VADDR", 35), ("SRSRC", 12), ("DMASK", 0x1), ("DIM", 1), ("UNORM", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    st.sgpr[24..28].copy_from_slice(&[0, 0, 0, 0]);
    st.sgpr[28..32].copy_from_slice(&[0, 0, 0, 1]);
    st.sgpr[32..40].copy_from_slice(&tsharp(2, 20, 9));
    st.sgpr[40..48].copy_from_slice(&tsharp(4, 20, 9));
    st.sgpr[48..56].copy_from_slice(&tsharp(6, 20, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = f((x as f32 + 0.5) / 4.0);
        r[1] = f((y as f32 + 0.5) / 4.0);
        r[3] = x;
        r[4] = y;
        r[5] = 0;
        r[6] = if l < 16 { 10 * l as u32 } else { 0 };
        r[7] = l as u32;
        r[30] = f(0.5); // between texel columns 1 and 2
        r[31] = f(0.125); // row 0
        r[32] = f(0.0); // lod
        r[33] = f(0.5);
        r[34] = f(0.5);
        r[35] = 0;
        r[36] = 0;
        r[37] = x; // lanes 16.. land outside the 4x4 texture: their stores are dropped
        r[38] = l as u32 / 4;
        r[39] = 100;
    }
    // tex0: 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 4, height: 4, depth: 1, texel_bytes: 4, data: vec![0xEE; 64], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 4], uav: true },
        TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32_UINT, width: 1, height: 1, depth: 1, texel_bytes: 4, data: vec![0; 4], uav: true },
    ];
    let samp = vec![
        TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
        TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None },
    ];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [x, y, x + 4.0 * y, 1.0], "lane {l} sample_lz point");
        assert_eq!([fl(20), fl(21), fl(22), fl(23)], [1.5, 0.0, 1.5, 1.0], "lane {l} sample_l linear");
        assert_eq!([fl(40), fl(41), fl(42), fl(43)], [1.0, 2.0, 2.0, 1.0], "lane {l} gather4 red");
        assert_eq!([fl(14), fl(15)], [x, x + 4.0 * y], "lane {l} image_load dmask x,z");
        assert_eq!(&v[50..54], &[4, 4, 1, 1], "lane {l} get_resinfo");
    }
    let t1: Vec<u32> = r.textures[0].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for i in 0..16 {
        assert_eq!(t1[i], 10 * i as u32, "image_store texel {i}");
    }
    assert_eq!(u32::from_le_bytes(r.textures[1][..4].try_into().unwrap()), 63, "image_atomic_umax");
    assert_eq!(u32::from_le_bytes(r.textures[2][..4].try_into().unwrap()), 64, "image_atomic_inc (64 lanes, limit 100)");
}

/// Static samplers: the program builds two S# from constants (point / linear, clamp) – as
/// the console compiler places a root signature's static samplers – and samples the same
/// texel boundary with each; they become static samplers of the root signature.
#[test]
fn gpu_static_samplers_from_constants_in_code() {
    if !gpu_ready_wave32() {
        return;
    }
    use runner::{TestTexture, TexDim};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let clamp = 2 | 2 << 3 | 2 << 6;
    let point = [clamp, 0x00FF_F000u32, 0, 0];
    let linear = [clamp, 0x00FF_F000, 1 << 20 | 1 << 22, 0];
    let mov = |r: u64, v: u32| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        if v == 0 {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 128)], None)
        } else {
            ("S_MOV_B32", "ENC_SOP1", vec![("SDST", r), ("SSRC0", 255)], Some(v))
        }
    };
    let mut list = Vec::new();
    for k in 0..4 {
        list.push(mov(24 + k as u64, point[k]));
        list.push(mov(28 + k as u64, linear[k]));
    }
    // T# tex0 in s[16:23] (SRSRC = 4); S# point s[24:27] (SSAMP = 6), linear s[28:31] (7)
    list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1)], None));
    list.push(("IMAGE_SAMPLE_LZ", "ENC_MIMG", vec![("VDATA", 20), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 7), ("DMASK", 0xF), ("DIM", 1)], None));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = translate(&code, &Options::default()).unwrap();
    assert_eq!(t.static_samplers, vec![(point, false), (linear, false)], "two static samplers from the code");
    assert!(!t.hlsl.contains("SamplerDescriptorHeap"), "no sampler heap lookups left");

    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for r in st.vgpr.iter_mut() {
        r[0] = f(0.5); // the boundary between texel columns 1 and 2
        r[1] = f(0.125); // row 0
    }
    // 4x4 RGBA32F, texel (x, y) = (x, y, x + 4y, 1)
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &[]).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let fl = |i: usize| f32::from_bits(v[i]);
        assert_eq!([fl(10), fl(11), fl(12), fl(13)], [2.0, 0.0, 2.0, 1.0], "lane {l}: static point sampler");
        assert_eq!([fl(20), fl(21), fl(22), fl(23)], [1.5, 0.0, 1.5, 1.0], "lane {l}: static linear sampler");
    }
}

#[test]
fn gpu_images_a16_d16() {
    if !gpu_ready_wave32() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mimg = |name: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(name, "ENC_MIMG", f, None).unwrap() };
    let mut code = Vec::new();
    // coordinates (u, v) as f16 in v0; result as 4 halves in v[10:11]
    code.extend(mimg("IMAGE_SAMPLE_LZ", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0xF), ("DIM", 1), ("A16", 1), ("D16", 1)]));
    // integer coordinates (x, y) as 16-bit in v1; x and z in v[12:13] as 32-bit
    code.extend(mimg("IMAGE_LOAD", &[("VDATA", 12), ("VADDR", 1), ("SRSRC", 4), ("DMASK", 0x5), ("DIM", 1), ("UNORM", 1), ("A16", 1)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    let h = |x: f32| half_bits(x);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as u32, (l / 4 % 4) as u32);
        r[0] = h((x as f32 + 0.5) / 4.0) | h((y as f32 + 0.5) / 4.0) << 16;
        r[1] = x | y << 16;
    }
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, (x + 4 * y) as f32, 1.0] {
                t0.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let samp = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let v = &r.vgpr[l];
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        assert_eq!((v[10], v[11]), (h(x) | h(y) << 16, h(x + 4.0 * y) | h(1.0) << 16), "lane {l} a16+d16 sample");
        assert_eq!([f32::from_bits(v[12]), f32::from_bits(v[13])], [x, x + 4.0 * y], "lane {l} a16 load");
    }
}

/// f32 → IEEE half bits (exact for the small values used in tests).
fn half_bits(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    (b >> 16 & 0x8000) | ((e as u32) << 10) | ((b >> 13) & 0x3FF)
}

#[test]
fn gpu_format_stores_atomics_global_lds() {
    if !gpu_ready() {
        return;
    }
    let mubuf = |n: &'static str, f: &[(&str, u64)]| -> Vec<u8> { encode(n, "ENC_MUBUF", f, None).unwrap() };
    let mut code = Vec::new();
    // V# A (s[12:15]): 8_8_8_8_UNORM at 0x100, stride 4 (idxen with v0 = lane)
    code.extend(mubuf("BUFFER_STORE_FORMAT_XYZW", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)]));
    // V# B (s[16:19]): 16_FLOAT at 0x300, stride 2 → neighbouring lanes share dwords
    code.extend(mubuf("BUFFER_STORE_FORMAT_X", &[("VDATA", 14), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)]));
    // V# C (s[20:23]): raw at 0x400: every lane adds 1, umax returns the old value
    code.extend(mubuf("BUFFER_ATOMIC_ADD", &[("VDATA", 15), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128)]));
    code.extend(mubuf("BUFFER_ATOMIC_UMAX", &[("VDATA", 16), ("VADDR", 0), ("SRSRC", 5), ("SOFFSET", 128), ("OFFSET", 4), ("GLC", 1)]));
    // global: store lane*3 to s[2:3] + v1 (lane*4), load it back through the 64-bit form
    code.extend(encode("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 17), ("SADDR", 2), ("OFFSET", 0)], None).unwrap());
    code.extend(encode("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 18), ("ADDR", 20), ("SADDR", 0x7D), ("OFFSET", 0xFFC)], None).unwrap()); // offset -4
    // LDS: write lane at lane*4, barrier, read reversed, read2 of two neighbours, counter, bpermute
    code.extend(encode("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 0)], None).unwrap());
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(encode("DS_READ_B32", "ENC_DS", &[("VDST", 22), ("ADDR", 23)], None).unwrap());
    code.extend(encode("DS_READ2_B32", "ENC_DS", &[("VDST", 24), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("DS_ADD_RTN_U32", "ENC_DS", &[("VDST", 26), ("ADDR", 27), ("DATA0", 28), ("OFFSET0", 0), ("OFFSET1", 2)], None).unwrap()); // at 512
    code.extend(encode("DS_BPERMUTE_B32", "ENC_DS", &[("VDST", 29), ("ADDR", 23), ("DATA0", 0)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x500; // global base
    st.sgpr[12..16].copy_from_slice(&[0x100, 4 << 16, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 56 << 12]);
    st.sgpr[16..20].copy_from_slice(&[0x300, 2 << 16, 64, (4 | 5 << 3 | 6 << 6 | 7 << 9) | 13 << 12]);
    st.sgpr[20..24].copy_from_slice(&[0x400, 0, 16, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = l32;
        r[1] = 4 * l32;
        r[10] = f(1.0);
        r[11] = f(0.0);
        r[12] = f(l as f32 / 63.0);
        r[13] = f(0.5);
        r[14] = f(l as f32);
        r[15] = 1;
        r[16] = 100 + l32;
        r[17] = 3 * l32;
        r[20] = 0x500 + 4 * l32 + 4; // 64-bit address (hi in v21 = 0), offset -4
        r[23] = 4 * (63 - l32);
        r[27] = 0;
        r[28] = 1;
    }
    let r = runner::run_code(&code, &st, &vec![0u32; 1024]).unwrap();
    let byte = |a: usize| (r.mem[a / 4] >> (8 * (a % 4))) & 0xFF;
    for l in 0..64usize {
        let px = [byte(0x100 + 4 * l), byte(0x100 + 4 * l + 1), byte(0x100 + 4 * l + 2), byte(0x100 + 4 * l + 3)];
        assert_eq!(px, [255, 0, ((l as f32 / 63.0) * 255.0).round() as u32, 128], "lane {l} store 8_8_8_8_UNORM");
        let h = (r.mem[(0x300 + 2 * l) / 4] >> (16 * (l % 2))) & 0xFFFF;
        assert_eq!(h, half_bits(l as f32), "lane {l} store 16_FLOAT (shared dwords)");
        let v = &r.vgpr[l];
        assert_eq!(r.mem[0x500 / 4 + l], 3 * l as u32, "lane {l} global_store saddr");
        assert_eq!(v[18], 3 * l as u32, "lane {l} global_load 64-bit address, offset -4");
        assert_eq!(v[22], 63 - l as u32, "lane {l} LDS reversed");
        assert_eq!(v[24], l as u32, "lane {l} ds_read2 (first)");
        if l < 63 {
            assert_eq!(v[25], l as u32 + 1, "lane {l} ds_read2 (second; LDS is not zeroed past the written area)");
        }
        assert_eq!(v[29], 63 - l as u32, "lane {l} ds_bpermute");
        assert!(v[16] < 164, "lane {l} umax old value");
    }
    assert_eq!(r.mem[0x400 / 4], 64, "buffer_atomic_add");
    assert_eq!(r.mem[0x404 / 4], 163, "buffer_atomic_umax");
    let mut olds: Vec<u32> = r.vgpr.iter().map(|v| v[26]).collect();
    olds.sort();
    // LDS is not zeroed between dispatches (nor on the console): consecutive values from any start
    let base = olds[0];
    assert_eq!(olds, (0..64).map(|k| base.wrapping_add(k)).collect::<Vec<u32>>(), "ds_add_rtn_u32 returns every old value once");
}

/// global_*_dword_addtid (ENC_FLAT_GLBL): the spec gives them no ADDR operand – the
/// address is the scalar SADDR pair ("off"/null encodes a zero base) plus 4 times the
/// lane index plus the signed 12-bit OFFSET. Emitted HLSL is deterministic.
#[test]
fn global_dword_addtid_address_is_saddr_plus_4_per_lane() {
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mut code = Vec::new();
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 5), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 7), ("SADDR", 0x7D), ("OFFSET", 0xFFC)])); // null base, offset -4
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let t = translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}\n{}", t.unsupported_why, t.hlsl);
    // s[2:3] base, signed +8 offset; the ADDR field bits are never read as a VGPR
    assert!(t.hlsl.contains("uint t1_a = s2 + 4u * lane_id + 0x00000008u;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("mem.Store((t1_a + 0u) & ~3u, v5);"), "{}", t.hlsl);
    // "off"/null SADDR → base 0; the 12-bit offset sign-extends (0xFFC → -4)
    assert!(t.hlsl.contains("uint t2_a = 4u * lane_id + 0xFFFFFFFCu;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("v7 = mem.Load((t2_a + 0u) & ~3u);"), "{}", t.hlsl);
    // exactly the two address expressions above – no ADDR VGPR read anywhere
    assert_eq!(t.hlsl.matches("_a = ").count(), 2, "{}", t.hlsl);
    if super::dxc::location().is_ok() {
        super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{}", t.hlsl));
    }
}

/// On hardware: every lane touches its own dword (EA = SADDR + 4·lane + OFFSET), the
/// "off" SADDR sentinel gives a zero base and the signed offset can point below the base.
#[test]
fn gpu_dword_addtid_per_lane_addresses() {
    if !gpu_ready_wave32() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mut code = Vec::new();
    // each lane writes its signature at s[2:3] + 4·lane + 8, then reads it back (+8) and
    // the previous lane's dword (+4)
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 5), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 7), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 8), ("SADDR", 2), ("OFFSET", 4)]));
    // s[4:5] + 4·lane − 4 → preset values just below 0x200; offset −4
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 10), ("SADDR", 4), ("OFFSET", 0xFFC)]));
    // null ("off") SADDR: base 0 → per-lane dwords at 0x40 + 4·lane
    code.extend(glbl("GLOBAL_STORE_DWORD_ADDTID", &[("DATA", 9), ("SADDR", 0x7D), ("OFFSET", 0x40)]));
    code.extend(glbl("GLOBAL_LOAD_DWORD_ADDTID", &[("VDST", 11), ("SADDR", 0x7D), ("OFFSET", 0x40)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[4] = 0x200;
    let sig = |l: usize| 0xA500 + l as u32;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[5] = sig(l);
        r[9] = 0xBEEF + l as u32;
    }
    let mut mem = vec![0u32; 512];
    mem[0x604 / 4] = 0xDEAD_CAFE; // in the gap below the stores: lane 0's +4 load lands here
    for l in 0..64usize {
        mem[0x1FC / 4 + l] = 0xF00 + l as u32; // under s[4:5] for the offset −4 loads
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for l in 0..64usize {
        assert_eq!(r.mem[0x608 / 4 + l], sig(l), "lane {l}: store at saddr + 4·lane + 8");
        assert_eq!(r.vgpr[l][7], sig(l), "lane {l}: load its own dword");
        let prev = if l == 0 { 0xDEAD_CAFE } else { sig(l - 1) };
        assert_eq!(r.vgpr[l][8], prev, "lane {l}: offset +4 reads the previous lane's dword");
        assert_eq!(r.vgpr[l][10], 0xF00 + l as u32, "lane {l}: signed offset −4 below the base");
        assert_eq!(r.mem[0x40 / 4 + l], 0xBEEF + l as u32, "lane {l}: store with a null SADDR base");
        assert_eq!(r.vgpr[l][11], 0xBEEF + l as u32, "lane {l}: load with a null SADDR base");
    }
}

#[test]
fn gpu_valu_16_and_64_bit() {
    if !gpu_ready_wave32() {
        return;
    }
    let p = prog(&[
        ("V_ADD_F16", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)], None), // low halves, high kept
        ("V_MUL_F16", "ENC_VOP3", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL", 0b1001)], None), // src0.hi * src1.lo → dst.hi
        ("V_LSHRREV_B16", "ENC_VOP3", &[("VDST", 4), ("SRC0", 132), ("SRC1", V + 5)], None),
        ("V_CMP_LT_F16", "ENC_VOPC", &[("SRC0", V + 6), ("VSRC1", 7)], None),
        ("V_LSHLREV_B64", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 10), ("SRC1", V + 12)], None),
        ("V_CMP_EQ_U64", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 8), ("SRC1", V + 14)], None),
        ("V_MAD_U64_U32", "VOP3_SDST_ENC", &[("VDST", 16), ("SDST", 22), ("SRC0", V + 18), ("SRC1", V + 19), ("SRC2", V + 24)], None),
        ("V_ADD_F64", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 26), ("SRC1", 242)], None), // + 1.0 (double inline constant)
        ("V_CVT_F32_F64", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 26)], None),
        ("V_CEIL_F64", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 40)], None),
        ("V_FRACT_F64", "ENC_VOP1", &[("VDST", 32), ("SRC0", V + 40)], None),
        ("V_SQRT_F64", "ENC_VOP1", &[("VDST", 34), ("SRC0", V + 40)], None),
        ("V_RSQ_F64", "ENC_VOP1", &[("VDST", 36), ("SRC0", V + 40)], None),
    ]);
    let h = half_bits;
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[0] = h(1.5) | h(-2.0) << 16;
        r[1] = h(2.25) | h(0.5) << 16;
        r[2] = 0xBEEF_1234;
        r[3] = 0x0000_1111;
        r[4] = 0xCAFE_0000;
        r[5] = 0xFFFF_8421;
        r[6] = h(l as f32);
        r[7] = h(31.5);
        r[10] = l32;
        r[12] = 1;
        let e = if l % 2 == 0 { 1u64 << l } else { 0 };
        r[14] = e as u32;
        r[15] = (e >> 32) as u32;
        r[18] = 0xFFFF_FFFF;
        r[19] = l32 + 1;
        let c = if l % 2 == 1 { u64::MAX } else { l as u64 };
        r[24] = c as u32;
        r[25] = (c >> 32) as u32;
        let dv = (l as f64 + 0.25).to_bits();
        r[26] = dv as u32;
        r[27] = (dv >> 32) as u32;
        let x = (l as f64 + 1.25).powi(2).to_bits();
        r[40] = x as u32;
        r[41] = (x >> 32) as u32;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[2], 0xBEEF_0000 | h(3.75), "lane {l} v_add_f16 keeps the high half");
        assert_eq!(v[3], 0x1111 | h(-4.5) << 16, "lane {l} v_mul_f16 op_sel");
        assert_eq!(v[4], 0xCAFE_0842, "lane {l} v_lshrrev_b16");
        assert_eq!(v[8] as u64 | (v[9] as u64) << 32, 1u64 << l, "lane {l} v_lshlrev_b64");
        let c = if l % 2 == 1 { u64::MAX } else { l as u64 };
        let (w, carry) = (0xFFFF_FFFFu64 * (l as u64 + 1)).overflowing_add(c);
        assert_eq!(v[16] as u64 | (v[17] as u64) << 32, w, "lane {l} v_mad_u64_u32");
        let _ = carry;
        assert_eq!(f64::from_bits(v[26] as u64 | (v[27] as u64) << 32), l as f64 + 1.25, "lane {l} v_add_f64");
        assert_eq!(f32::from_bits(v[28]), l as f32 + 1.25, "lane {l} v_cvt_f32_f64");
        let d = |n: usize| f64::from_bits(v[n] as u64 | (v[n + 1] as u64) << 32);
        let root = l as f64 + 1.25;
        let input = root * root;
        assert_eq!((d(30), d(32), d(34)), (input.ceil(), input.fract(), root), "lane {l} unary f64");
        assert!((d(36) - root.recip()).abs() < 1e-15, "lane {l} v_rsq_f64");
    }
    assert_eq!((r.sgpr[106], r.sgpr[107]), (0xFFFF_FFFF, 0), "v_cmp_lt_f16 → vcc");
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0x5555_5555, 0x5555_5555), "v_cmp_eq_u64 → sdst");
    let carries: u64 = (0..64).filter(|&l| (0xFFFF_FFFFu64 * (l as u64 + 1)).overflowing_add(if l % 2 == 1 { u64::MAX } else { l as u64 }).1).map(|l| 1u64 << l).sum();
    assert_eq!(r.sgpr[22] as u64 | (r.sgpr[23] as u64) << 32, carries, "v_mad_u64_u32 carry mask");
}

#[test]
fn gpu_packed_math_dots_movrel() {
    if !gpu_ready_wave32() {
        return;
    }
    let p = prog(&[
        ("V_PK_ADD_F16", "ENC_VOP3P", &[("VDST", 2), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        // lo = src0.hi * src1.lo, hi = -(src0.lo * src1.hi)
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 3), ("SRC0", V), ("SRC1", V + 1), ("OP_SEL", 1), ("OP_SEL_HI", 2), ("NEG_HI", 1)], None),
        ("V_DOT2_F32_F16", "ENC_VOP3P", &[("VDST", 4), ("SRC0", V), ("SRC1", V + 1), ("SRC2", V + 5), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4_U32_U8", "ENC_VOP3P", &[("VDST", 6), ("SRC0", V + 7), ("SRC1", V + 8), ("SRC2", V + 9), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_DOT4_I32_I8", "ENC_VOP3P", &[("VDST", 11), ("SRC0", V + 7), ("SRC1", V + 12), ("SRC2", V + 9), ("OP_SEL_HI", 3), ("OP_SEL_HI_2", 1)], None),
        ("V_PK_FMAC_F16", "ENC_VOP2", &[("VDST", 13), ("SRC0", V), ("VSRC1", 1)], None),
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 124), ("SSRC0", 131)], None), // m0 = 3
        ("V_MOVRELS_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 20)], None), // v10 = v23
        ("V_MOVRELD_B32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 9)], None), // v33 = v9
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 40), ("SRC0", V + 33)], None),
    ]);
    let h = half_bits;
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = h(1.5) | h(-2.0) << 16;
        r[1] = h(2.25) | h(0.5) << 16;
        r[5] = f(10.0);
        r[7] = 0x0102_0304;
        r[8] = 0x0506_0708;
        r[9] = l as u32;
        r[12] = 0xFF06_0708; // -1, 6, 7, 8
        r[13] = h(1.0) | h(2.0) << 16;
        for k in 0..8 {
            r[20 + k] = 100 + k as u32;
        }
        r[33] = 0xDEAD;
    }
    let r = runner::run_program(&p, &st, &[]).unwrap().0;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        assert_eq!(v[2], h(3.75) | h(-1.5) << 16, "lane {l} v_pk_add_f16");
        assert_eq!(v[3], h(-4.5) | h(-0.75) << 16, "lane {l} v_pk_mul_f16 op_sel / neg_hi");
        assert_eq!(f32::from_bits(v[4]), 12.375, "lane {l} v_dot2_f32_f16");
        assert_eq!(v[6], 70 + l as u32, "lane {l} v_dot4_u32_u8");
        assert_eq!(v[11] as i32, 4 * 8 + 3 * 7 + 2 * 6 - 1 + l as i32, "lane {l} v_dot4_i32_i8 (bytes from the low end)");
        assert_eq!(v[13], h(1.0 + 1.5 * 2.25) | h(2.0 - 2.0 * 0.5) << 16, "lane {l} v_pk_fmac_f16");
        assert_eq!(v[10], 103, "lane {l} v_movrels_b32");
        assert_eq!(v[40], l as u32, "lane {l} v_movreld_b32");
    }
}

#[test]
fn dxc_mesh_shader_probe() {
    use super::dxc;
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return;
    }
    let src = r#"
struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };
struct P { bool cull : SV_CullPrimitive; };
groupshared uint xcl_nv;
[WaveSize(64)]
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(uint gi : SV_GroupIndex, out vertices V verts[128], out indices uint3 tris[128], out primitives P prims[128]) {
    if (gi == 0) xcl_nv = 3;
    GroupMemoryBarrierWithGroupSync();
    uint n = xcl_nv;
    SetMeshOutputCounts(n, 1);
    if (gi < n) { verts[gi].pos = float4(gi, 0, 0, 1); verts[gi].p0 = 0; }
    if (gi < 1) { tris[gi] = uint3(0, 1, 2); prims[gi].cull = false; }
}
"#;
    for prof in ["ms_6_6", "ms_6_8"] {
        let r = dxc::compile(src, "main", prof, &[]);
        println!("{prof}: {:?}", r.as_ref().map(|b| b.len()).map_err(|e| e.lines().take(3).collect::<Vec<_>>().join(" | ")));
    }
}

/// Mesh mode end to end: an NGG program in the console's style (wave index from s3, group
/// from v5 >> 7, GS_ALLOC_REQ, exp pos0/param0/prim, two waves per group) runs as a compute
/// shader, the pass-through mesh shader rasterises its records.
#[test]
fn gpu_mesh_mode_two_passes() {
    if !gpu_ready() || !mesh_ready() {
        return;
    }
    use super::{dxc, mesh_passthrough, MsInterface, MESH_HEADER};
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let ms = MsInterface { outputs: vec![sig("SV_Position", 1, 0), sig("TEXCOORD", 0, 1)], threads: 128, max_vertices: 128, max_primitives: 128 };
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None)
    };
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)), // wave index
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None), // thread in group
        ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", vec![("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None), // group = v5 >> 7
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None), // x += group
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 17), ("SRC0", 242)], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 15, 16, 17], 0));
    list.push(("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None)); // thread 0
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)));
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let code = prog(&refs);
    let t = translate(&code, &Options { mode: Mode::Mesh(ms.clone()), wave64: true, ..Default::default() }).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let pass = mesh_passthrough(&ms);
    let msb = dxc::compile(&pass, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass}"));
    let ps_src = "struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();
    // vertex table at byte 0x100, V# in s[16:19] (stride 16, 3 records)
    let mut mem = vec![0u32; 256];
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[f(*x), f(*y), f(0.0), f(1.0)]);
    }
    let mut sg = vec![0u32; 128];
    sg[16..20].copy_from_slice(&[0x100, 16 << 16, 3, 0]);
    let (px, rec) = runner::mesh_draw(&cs, &msb, &ps, 2, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    let (rd, vd) = (ms.record_dwords(), ms.vert_dwords());
    for g in 0..2usize {
        let r = &rec[g * rd..(g + 1) * rd];
        assert_eq!((r[0], r[1]), (3, 1), "group {g}: GS_ALLOC_REQ counts");
        for v in 0..3usize {
            let b = MESH_HEADER + v * vd;
            let x = [-1.0f32, -1.0, 0.0][v] + g as f32;
            assert_eq!(f32::from_bits(r[b]), x, "group {g} vertex {v} pos.x");
            assert_eq!([f32::from_bits(r[b + 4]), f32::from_bits(r[b + 5]), f32::from_bits(r[b + 7])], [g as f32, v as f32, 1.0], "group {g} vertex {v} param0");
        }
        assert_eq!(r[MESH_HEADER + 128 * vd], 1 << 10 | 2 << 20, "group {g} primitive");
    }
    let at = |x: usize, y: usize| px[y * 32 + x];
    assert_eq!((at(3, 29)[0], at(3, 29)[3]), (0.0, 1.0), "group 0 triangle (left)");
    assert_eq!((at(19, 29)[0], at(19, 29)[3]), (1.0, 1.0), "group 1 triangle (right)");
    assert_eq!(at(29, 3), [-1.0; 4], "outside both triangles");

    // the same records through a pass-through for a pixel shader with 16 inputs: over the
    // mesh output limit (17 · 16 B · 128 vertices), so 4 mesh groups per record, each a run
    // of triangles with their own vertex copies; inputs the program does not write read 0
    let ins: Vec<crate::gpu::dxbc::SigElement> = (0..16).map(|k| crate::gpu::dxbc::SigElement { index: k, register: k + 1, ..sig("TEXCOORD", 0, 0) }).collect();
    let split = super::emit::passthrough_split(&ms, super::emit::passthrough_elements(&ms, Some(&ins)));
    assert_eq!(split, 4);
    let pass16 = super::emit::mesh_passthrough_for(&ms, Some(&ins));
    let msb16 = dxc::compile(&pass16, "main", "ms_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{pass16}"));
    let fields: String = (0..16).map(|k| format!(" float4 t{k} : TEXCOORD{k};")).collect();
    let ps16 = dxc::compile(&format!("struct V {{ float4 pos : SV_Position;{fields} }};\nfloat4 main(V v) : SV_Target0 {{ return v.t0 + v.t15; }}\n"), "main", "ps_6_6", &[]).unwrap();
    let (px16, _) = runner::mesh_draw_split(&cs, &msb16, &ps16, 2, split, ms.record_dwords(), &sg, &mem, 32, [-1.0; 4]).unwrap();
    assert!(px16 == px, "split pass-through draws the same pixels");
}

/// DXC optimisation levels on a saved HLSL file (`XCL_TIME_HLSL=path`, profile in `XCL_TIME_PROFILE`).
#[test]
#[ignore]
fn dxc_time_levels() {
    use super::dxc;
    let Ok(path) = std::env::var("XCL_TIME_HLSL") else { return };
    let src = std::fs::read_to_string(path).unwrap();
    let prof = std::env::var("XCL_TIME_PROFILE").unwrap_or("cs_6_6".into());
    for args in [vec!["-O0"], vec!["-O1"], vec!["-O2"], vec!["-O3"]] {
        let t0 = std::time::Instant::now();
        let r = dxc::compile(&src, "main", &prof, &args);
        println!("{args:?}: {:.1} s, {:?}", t0.elapsed().as_secs_f64(), r.map(|b| b.len()).map_err(|e| e.chars().take(200).collect::<String>()));
    }
}

/// Minimal RGB8 PNG writer (zlib via miniz_oxide).
fn write_png(path: &std::path::Path, w: u32, h: u32, rgb: &[u8]) {
    fn crc(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }
    let chunk = |out: &mut Vec<u8>, ty: &[u8], d: &[u8]| {
        out.extend_from_slice(&(d.len() as u32).to_be_bytes());
        let mut td = ty.to_vec();
        td.extend_from_slice(d);
        out.extend_from_slice(&td);
        out.extend_from_slice(&crc(&td).to_be_bytes());
    };
    let mut raw = Vec::with_capacity((w * 3 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w as usize * 3..(y + 1) * w as usize * 3]);
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6));
    chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out).unwrap();
}

/// Retail pixel shaders (no textures) translated and drawn full screen on the GPU:
/// attributes = screen uv, constants from a pattern. `XCL_SAMPLE_RPF`, `XCL_GALLERY_DIR`.
#[test]
#[ignore]
fn retail_ps_gallery() {
    use super::{dxc, ranked};
    let (Ok(path), Ok(dir)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_GALLERY_DIR")) else { return };
    if !gpu_ready() {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data = std::fs::read(&path).unwrap();
    // descriptor table at 0x100: every V# points at the constants at 0x4000
    let mut mem = vec![0u32; 0x3000];
    for i in 0..256 {
        mem[0x40 + 4 * i..0x40 + 4 * i + 4].copy_from_slice(&[0x4000, 0, 0x8000, 0x27FAC]);
    }
    for i in 0..0x2000 {
        mem[0x1000 + i] = f(0.5 + 0.45 * ((i as f32) * 0.37).sin());
    }
    let mut io = vec![0u32; 128];
    for k in (0..106).step_by(2) {
        io[k] = 0x100;
    }
    let size = 160u32;
    let t0 = std::time::Instant::now();
    let (mut tried, mut kept) = (0, Vec::new());
    for rec in crate::gpu::pack::scan(&data, 0x40) {
        if kept.len() >= 24 || t0.elapsed().as_secs() > 150 {
            break;
        }
        if !rec.name.starts_with("PS_") {
            continue;
        }
        let Some(blob) = crate::gpu::pack::inflate(&data, &rec) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        let Some(fs) = sh.fssm.first() else { continue };
        let Some(p) = fs.programs.first() else { continue };
        let Some(mode) = super::interface(&blob, &sh, fs) else { continue };
        let Mode::Pixel(ps) = mode.clone() else { continue };
        if !ps.outputs.iter().any(|e| e.system_value == 64 && e.index == 0) {
            continue;
        }
        let code = &fs.data[p.offset..p.offset + p.len];
        let Ok(t) = translate(code, &Options { mode, wave64: super::detect_wave64(code), ..Default::default() }) else { continue };
        if !t.complete() || t.hlsl.contains("ResourceDescriptorHeap") || t.hlsl.len() > 150_000 {
            continue;
        }
        tried += 1;
        let mut vs = String::from("struct O {\n    float4 pos : SV_Position;\n");
        let ins = ranked(&ps.inputs, false);
        for (k, e) in ins.iter().enumerate() {
            let ty = match e.component_type {
                1 => "uint",
                2 => "int",
                _ => "float",
            };
            vs.push_str(&format!("    {ty}4 a{k} : {};\n", e.semantic()));
        }
        vs.push_str("};\nO main(uint id : SV_VertexID) {\n    O o;\n    float2 uv = float2((id << 1) & 2, id & 2);\n");
        vs.push_str("    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0.5, 1);\n");
        for (k, e) in ins.iter().enumerate() {
            let v = match e.component_type {
                1 => "uint4(uv * 255.0, 0, 1)",
                2 => "int4(uv * 255.0, 0, 1)",
                _ => "float4(uv, 1.0 - uv.x, 1.0)",
            };
            vs.push_str(&format!("    o.a{k} = {v};\n"));
        }
        vs.push_str("    return o;\n}\n");
        let (Ok(vsb), Ok(psb)) = (dxc::compile(&vs, "main", "vs_6_6", &[]), dxc::compile(&t.hlsl, "main", "ps_6_6", &[])) else { continue };
        let px = match runner::draw(&vsb, &psb, &[], &[0.0f32; 3], 4, &io, &mem, size, [0.0, 0.0, 0.0, 1.0]) {
            Ok(px) => px,
            Err(e) => {
                println!("{}: {e}", rec.name);
                break; // device trouble: stop
            }
        };
        let rgb: Vec<f32> = px.iter().flat_map(|c| [c[0], c[1], c[2]]).map(|x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
        let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
        let var = rgb.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / rgb.len() as f32;
        if var < 0.002 {
            continue; // flat colour: not worth a picture
        }
        let bytes: Vec<u8> = rgb.iter().map(|x| (x.powf(1.0 / 2.2) * 255.0).round() as u8).collect();
        let file = format!("{:02}_{}.png", kept.len(), rec.name.split('_').take(4).collect::<Vec<_>>().join("_"));
        write_png(&dir.join(&file), size, size, &bytes);
        println!("{file}: {} instructions, variance {var:.3}", t.instructions);
        kept.push(format!("{file}\t{}\t{}", rec.name, t.instructions));
    }
    std::fs::write(dir.join("index.txt"), kept.join("\n")).unwrap();
    println!("tried {tried}, saved {} in {:.0} s", kept.len(), t0.elapsed().as_secs_f64());
}

#[test]
fn gpu_image_sample_offset() {
    if !gpu_ready_wave32() {
        return;
    }
    use runner::{TestSampler, TestTexture, TexDim};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
    let mut code = encode("IMAGE_SAMPLE_LZ_O", "ENC_MIMG", &[("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SSAMP", 6), ("DMASK", 0x3), ("DIM", 1)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[16..24].copy_from_slice(&tsharp(0, 77, 9));
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        r[0] = 1 | 0x3F << 8; // offset (+1, -1)
        r[1] = f((x + 0.5) / 4.0);
        r[2] = f((y + 0.5) / 4.0);
    }
    let mut t0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for v in [x as f32, y as f32, 0.0, 1.0] {
                t0.extend_from_slice(&f32::to_le_bytes(v));
            }
        }
    }
    let tex = vec![TestTexture { dim: TexDim::D2, format: DXGI_FORMAT_R32G32B32A32_FLOAT, width: 4, height: 4, depth: 1, texel_bytes: 16, data: t0, uav: false }];
    let samp = vec![TestSampler { filter: D3D12_FILTER_MIN_MAG_MIP_POINT, address: D3D12_TEXTURE_ADDRESS_MODE_CLAMP, comparison: None }];
    let r = runner::run_code_with(&code, &st, &[], &tex, &samp).unwrap();
    for l in 0..64 {
        let (x, y) = ((l % 4) as f32, (l / 4 % 4) as f32);
        let got = [f32::from_bits(r.vgpr[l][10]), f32::from_bits(r.vgpr[l][11])];
        assert_eq!(got, [(x + 1.0).min(3.0), (y - 1.0).max(0.0)], "lane {l} sample_lz_o");
    }
}

/// Console unified memory: the GPU reads constants and buffers straight from process memory
/// (4 GiB-aligned arena opened as a D3D12 heap) and its stores land there, no copies.
#[test]
fn gpu_unified_memory_arena() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::new(1 << 20).unwrap();
    let va = arena.base() as u64;
    assert_eq!(va & 0xFFFF_FFFF, 0, "4 GiB alignment");
    let words = || unsafe { std::slice::from_raw_parts_mut(arena.base() as *mut u32, arena.size() / 4) };
    let w = words();
    w[0x1000 / 4] = 0xC0FFEE;
    w[0x1000 / 4 + 1] = 0x1234;
    for l in 0..64 {
        w[0x2000 / 4 + l] = 1000 + l as u32;
    }
    let p = prog(&[
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3] = console address
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    let t = translate(&p, &Options::default()).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let hi = (va >> 32) as u32;
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(va + 0x1000) as u32, hi]);
    st.sgpr[8..12].copy_from_slice(&[(va + 0x2000) as u32, hi & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[(va + 0x3000) as u32, hi & 0xFFFF | 4 << 16, 64, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = l as u32;
        r[2] = 7 * l as u32;
    }
    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0xC0FFEE, 0x1234), "s_load from process memory");
    for l in 0..64 {
        assert_eq!(r.vgpr[l][1], 1000 + l as u32, "lane {l} buffer_load from process memory");
        assert_eq!(words()[0x3000 / 4 + l], 7 * l as u32, "lane {l} buffer_store visible to the CPU");
    }
}

/// Memory above 4 GiB: two UMA windows 4 GiB apart; every access kind picks its window from
/// the high address bits (V#, SGPR pair, 64-bit VGPR address, SADDR pair) and reaches the
/// right one – s_load, buffer load / store, global load / store / atomic in window 1 next to
/// accesses in window 0.
#[test]
fn gpu_memory_windows_above_4_gib() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let word = |a: u64| unsafe { &mut *(a as *mut u32) };
    *word(w1 + 0x1000) = 0xC0FFEE;
    *word(w1 + 0x1004) = 0x1234;
    for l in 0..64u64 {
        *word(w1 + 0x2000 + 4 * l) = 1000 + l as u32;
        *word(w1 + 0x4000 + 4 * l) = 5000 + l as u32;
        *word(w0 + 0x4000 + 4 * l) = 9; // same low address in window 0: must not be read
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| -> (&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>) {
        (n, "ENC_FLAT_GLBL", f.iter().map(|&(a, b)| (unsafe { std::mem::transmute::<&str, &'static str>(a) }, b)).collect(), None)
    };
    let list = vec![
        ("S_LOAD_DWORDX2", "ENC_SMEM", vec![("SDATA", 20), ("SBASE", 1), ("OFFSET", 0)], None), // s[2:3] → window 1
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None), // V# in window 1
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None), // V# in window 0
        glbl("GLOBAL_LOAD_DWORD", &[("VDST", 6), ("ADDR", 4), ("SADDR", 0x7D), ("OFFSET", 0)]), // v[4:5] → window 1
        glbl("GLOBAL_STORE_DWORD", &[("ADDR", 8), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", 0)]), // v[8:9] → window 1
        glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 10), ("ADDR", 3), ("DATA", 11), ("SADDR", 16), ("OFFSET", 0), ("GLC", 1)]), // s[16:17] + v3 → window 1
    ];
    let refs: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d)).collect();
    let p = prog(&refs);
    let one = translate(&p, &Options::default()).unwrap();
    assert!(!one.hlsl.contains("mem1") && !one.hlsl.contains("xcl_w =") && !one.hlsl.contains("xcl_m"), "one window: the single-window HLSL");
    let t = translate(&p, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("RWByteAddressBuffer mem1 : register(u3)"));
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();
    let hi = |a: u64| (a >> 32) as u32;
    let mut st = KernelState::default();
    st.sgpr[2..4].copy_from_slice(&[(w1 + 0x1000) as u32, hi(w1)]);
    st.sgpr[8..12].copy_from_slice(&[(w1 + 0x2000) as u32, hi(w1) & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[12..16].copy_from_slice(&[(w0 + 0x3000) as u32, hi(w0) & 0xFFFF | 4 << 16, 64, 0]);
    st.sgpr[16..18].copy_from_slice(&[(w1 + 0x6000) as u32, hi(w1)]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l = l as u64;
        r[0] = l as u32;
        r[2] = 7 * l as u32;
        r[3] = 0; // atomic offset: all lanes on one counter
        r[4] = (w1 + 0x4000 + 4 * l) as u32;
        r[5] = hi(w1);
        r[8] = (w1 + 0x5000 + 4 * l) as u32;
        r[9] = hi(w1);
        r[11] = 1;
    }
    let r = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    assert_eq!((r.sgpr[20], r.sgpr[21]), (0xC0FFEE, 0x1234), "s_load from window 1");
    for l in 0..64u64 {
        assert_eq!(r.vgpr[l as usize][1], 1000 + l as u32, "lane {l}: buffer_load from window 1");
        assert_eq!(r.vgpr[l as usize][6], 5000 + l as u32, "lane {l}: global_load from window 1");
        assert_eq!(*word(w0 + 0x3000 + 4 * l), 7 * l as u32, "lane {l}: buffer_store into window 0");
        assert_eq!(*word(w1 + 0x5000 + 4 * l), 7 * l as u32, "lane {l}: global_store into window 1");
    }
    assert_eq!(*word(w1 + 0x6000), 64, "global_atomic_add in window 1");
    assert_eq!(*word(w0 + 0x6000), 0, "nothing in window 0 at the same low address");
}

/// End-to-end GPU test of the sparse-window VRAM runtime layout:
/// - Window 0..2 = sparse VRAM (tiled reserved buffers on DEFAULT heaps)
/// - Window 3 = committed coherent RAM (`OpenExistingHeapFromAddress`)
/// Exercises a DXC-compiled SM 6.6 translated RDNA2 compute shader (`mem_windows = 4`)
/// reading a 64-bit spilled root-argument constant from Window 3 (`s[0:1]`), loading two
/// V# buffer descriptors through a 32-bit `hi = 0` descriptor table pointer in Window 0
/// sparse VRAM (`console_heap`), reading input from `XMemAlloc` graphics memory in sparse
/// VRAM, writing to Window 3, and then mutating both the buffer and the descriptor slot in
/// Window 0 sparse VRAM between submits.
#[test]
fn gpu_sparse_vram_descriptors_buffers_and_xmem() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::*;

    if !gpu_ready() {
        return;
    }
    crate::hle::d3d12x::real::init().expect("real D3D12.X backend");
    let u = crate::gpu::uma::get().expect("UMA");
    assert_eq!(u.shader_windows(), 4, "runtime layout has 4 shader windows");
    assert!(u.is_sparse_window(0) && u.is_sparse_window(1) && u.is_sparse_window(2), "windows 0..2 sparse VRAM");
    assert!(u.is_committed_window(3), "window 3 coherent committed RAM");
    let windows = crate::hle::d3d12x::real::uma_window_vas();
    assert_eq!(windows.len(), 4);
    assert!(windows.iter().all(|&va| va != 0), "all 4 window GPU VAs must be valid: {windows:x?}");

    // 1. Console descriptor heap in Window 0 sparse VRAM (32-bit hi = 0 table addressing)
    let heap_obj = 0xDEAD_5EA7usize;
    assert!(crate::hle::d3d12x::console_heap::register(heap_obj, 4, 0x1000, 0x5000, 32, false, None, true));
    let heap_va = crate::hle::d3d12x::console_heap::start(heap_obj).unwrap() as usize;
    assert_eq!(u.window_of(heap_va), Some(0), "descriptor heap must live in window 0");
    assert!(u.is_sparse(heap_va), "descriptor heap must live in sparse VRAM");

    // 2. XMem graphics allocations in sparse VRAM
    let gfx_attrs = 1u64 << 29;
    let in_a = unsafe { crate::hle::kernelx::XMemAllocDefault(0x1_0000, gfx_attrs) } as usize;
    let in_b = unsafe { crate::hle::kernelx::XMemAllocDefault(0x1_0000, gfx_attrs) } as usize;
    assert!(in_a != 0 && in_b != 0 && u.is_sparse(in_a) && u.is_sparse(in_b), "XMem graphics buffers must live in sparse VRAM");

    // 3. Coherent Window 3 allocations: spilled root argument table + output buffer + io/res
    let arg_table = u.alloc(0x1_0000, 256).unwrap() as usize;
    let out_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    let io_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    let res_buf = u.alloc(0x1_0000, 256).unwrap() as usize;
    assert_eq!(u.window_of(arg_table), Some(3));
    assert_eq!(u.window_of(out_buf), Some(3));

    // Populate initial data & descriptors (slot 0 = V# in_a, slot 1 = V# out_buf)
    let vsharp = |addr: usize| -> [u32; 4] { [addr as u32, (((addr >> 32) as u32) & 0xFFFF) | (4 << 16), 64, 0] };
    unsafe {
        *(arg_table as *mut u32) = 77;
        for l in 0..64usize {
            *((in_a + 4 * l) as *mut u32) = 1000 + l as u32;
            *((in_b + 4 * l) as *mut u32) = 5000 + 2 * l as u32;
        }
    }
    let slot0 = crate::hle::d3d12x::console_heap::slot(heap_va as u64).unwrap();
    let slot1 = crate::hle::d3d12x::console_heap::slot(heap_va as u64 + 32).unwrap();
    crate::hle::d3d12x::console_heap::write(&slot0, &vsharp(in_a));
    crate::hle::d3d12x::console_heap::write(&slot1, &vsharp(out_buf));

    // Translate + DXC-compile RDNA2 compute shader (4 windows, 32-bit descriptor table with hi = 0)
    let p = prog(&[
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 20), ("SBASE", 0), ("OFFSET", 0)], None), // s[0:1] -> Window 3 root arg table
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)], None), // s5 = 0 (32-bit descriptor table pointer)
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0)], None), // s[4:5] + 0 -> Window 0 sparse desc slot 0
        ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 2), ("OFFSET", 32)], None), // s[4:5] + 32 -> Window 0 sparse desc slot 1
        ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 20), ("VSRC1", 1)], None),
        ("BUFFER_STORE_DWORD", "ENC_MUBUF", &[("VDATA", 2), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("IDXEN", 1)], None),
    ]);
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 6, ..Default::default() };
    let t = translate(&p, &Options { mode: Mode::Compute(cs), mem_windows: 4, ..Default::default() }).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();

    // Execute on the real D3D12.X device + queue (goes through q_execute_command_lists -> sparse::before_execute)
    unsafe {
        let dev_raw = crate::hle::d3d12x::real::device_raw();
        let dev = ID3D12Device::from_raw_borrowed(&dev_raw).expect("device").clone();
        let rs = crate::gpu::recomp::pso::abi_root_signature(&dev, false).expect("ABI root signature");
        let pso: ID3D12PipelineState = dev
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
                ..Default::default()
            })
            .expect("CreateComputePipelineState");
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).unwrap();
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();

        // User SGPRs: s[0:1] = 64-bit arg_table (Window 3), s4 = 32-bit descriptor table offset in Window 0
        let mut io = vec![0u32; crate::gpu::recomp::KERNEL_SGPRS];
        io[0] = arg_table as u32;
        io[1] = (arg_table >> 32) as u32;
        io[4] = (heap_va - u.base()) as u32;
        io[5] = 0xDEAD_BEEF; // shader overwrites s5 = 0 to prove hi == 0 32-bit addressing hits Window 0 sparse VRAM
        std::ptr::copy_nonoverlapping(io.as_ptr() as *const u8, io_buf as *mut u8, io.len() * 4);

        let record_and_run = |fence_val: u64, first: bool| {
            if !first {
                alloc.Reset().unwrap();
                list.Reset(&alloc, Some(&pso)).unwrap();
            }
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(0, crate::hle::d3d12x::real::uma_va_of(io_buf).unwrap());
            list.SetComputeRootUnorderedAccessView(1, crate::hle::d3d12x::real::uma_va_of(res_buf).unwrap());
            for (k, &va) in windows.iter().enumerate() {
                list.SetComputeRootUnorderedAccessView(2 + k as u32, va);
            }
            list.Dispatch(1, 1, 1);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, fence_val).unwrap();
            if fence.GetCompletedValue() < fence_val {
                let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
                fence.SetEventOnCompletion(fence_val, ev).unwrap();
                windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
                let _ = windows::Win32::Foundation::CloseHandle(ev);
            }
        };

        // Submit 1: reads slot 0 -> in_a (sparse VRAM) + arg_table constant 77 (Window 3)
        record_and_run(1, true);
        for l in 0..64usize {
            let got = *((out_buf + 4 * l) as *const u32);
            assert_eq!(got, 1000 + l as u32 + 77, "submit 1 lane {l} mismatch");
        }

        // Submit 2: mutate descriptor slot 0 in Window 0 sparse VRAM -> in_b, and mutate in_b[3] + arg_table
        *(arg_table as *mut u32) = 13;
        *((in_b + 4 * 3) as *mut u32) = 9999;
        crate::hle::d3d12x::console_heap::write(&slot0, &vsharp(in_b));
        record_and_run(2, false);
        for l in 0..64usize {
            let expected = if l == 3 { 9999 + 13 } else { 5000 + 2 * l as u32 + 13 };
            let got = *((out_buf + 4 * l) as *const u32);
            assert_eq!(got, expected, "submit 2 (mutated sparse descriptor + XMem buffer) lane {l} mismatch");
        }

        // Clean up
        crate::hle::d3d12x::console_heap::unregister(heap_obj);
        crate::hle::kernelx::XMemFreeDefault(in_a as *mut _, gfx_attrs);
        crate::hle::kernelx::XMemFreeDefault(in_b as *mut _, gfx_attrs);
        assert!(u.free(arg_table) && u.free(out_buf) && u.free(io_buf) && u.free(res_buf));
    }
}

/// Experiment: does OpenExistingHeapFromAddress accept a reservation with only part committed?
#[test]
#[ignore]
fn gpu_arena_partial_commit_probe() {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows_sys::Win32::System::Memory::*;
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev3: ID3D12Device3 = dev.unwrap().cast().unwrap();
        let size = 256usize << 20;
        let p = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE);
        let c = VirtualAlloc(p, 1 << 20, MEM_COMMIT, PAGE_READWRITE);
        println!("reserve {p:?} commit {c:?}");
        let r: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(p);
        println!("partial commit: {:?}", r.as_ref().map(|h| h.GetDesc().SizeInBytes));
        let q = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        let r2: windows::core::Result<ID3D12Heap> = dev3.OpenExistingHeapFromAddress(q);
        println!("full commit 256 MB: {:?}", r2.as_ref().map(|h| h.GetDesc().SizeInBytes));
        for mb in [1024usize, 2047, 2048, 3072, 4095] {
            let big = mb << 20;
            let b = VirtualAlloc(std::ptr::null(), big, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            let t0 = std::time::Instant::now();
            let r3: windows::core::Result<ID3D12Heap> = if b.is_null() { Err(windows::core::Error::from_thread()) } else { dev3.OpenExistingHeapFromAddress(b) };
            println!("full commit {mb} MB: {:?} in {:.2} s", r3.as_ref().map(|h| h.GetDesc().SizeInBytes), t0.elapsed().as_secs_f64());
            drop(r3);
            VirtualFree(b, 0, MEM_RELEASE);
        }
    }
}

/// Decoded GFX10 T#s give resource descriptions the driver accepts.
#[test]
fn gpu_tsharp_to_d3d12_resources() {
    if !gpu_ready() {
        return;
    }
    use crate::gpu::desc;
    let t = |fmt: u32, w: u32, h: u32, depth: u32, ty: u32, mips: u32| -> [u32; 8] {
        let (w1, h1) = (w - 1, h - 1);
        [0, fmt << 20 | (w1 & 3) << 30, (w1 >> 2) | h1 << 14, 4 | 5 << 3 | 6 << 6 | 7 << 9 | (mips - 1) << 16 | ty << 28, depth - 1, (mips - 1) << 4, 0, 0]
    };
    for (name, words) in [
        ("2D RGBA8 9 mips", t(56, 256, 256, 1, 9, 9)),
        ("2D BC7 sRGB", t(182, 128, 64, 1, 9, 1)),
        ("cube RGBA16F", t(71, 64, 64, 6, 11, 7)),
        ("3D R11G11B10", t(36, 32, 32, 16, 10, 1)),
        ("2D array BC1 ×8", t(169, 64, 64, 8, 13, 4)),
        ("1D R32F", t(22, 1024, 1, 1, 8, 1)),
    ] {
        let img = desc::image(&words).unwrap();
        let (res, _srv) = desc::image_d3d12(&img).unwrap();
        runner::try_create_texture(&res).unwrap_or_else(|e| panic!("{name}: {e} ({res:?})"));
    }
}

/// Prints signatures / PSV0 of named retail shaders (`XCL_SAMPLE_RPF`, `XCL_DUMP=a,b`).
#[test]
#[ignore]
fn dump_retail_interfaces() {
    let (Ok(path), Ok(names)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_DUMP")) else { return };
    let data = std::fs::read(path).unwrap();
    for r in crate::gpu::pack::scan(&data, 0x40) {
        if !names.split(',').any(|n| r.name.contains(n)) {
            continue;
        }
        let blob = crate::gpu::pack::inflate(&data, &r).unwrap();
        let (i, o) = crate::gpu::dxbc::signatures(&blob);
        let sh = crate::gpu::xbox::parse(&blob).unwrap();
        println!("{} hw {:?}", r.name, sh.fssm.iter().map(|f| f.stage_label()).collect::<Vec<_>>());
        println!("  ISG1: {:?}", i.iter().map(|e| format!("{}@r{}:{:x} sv{} t{}", e.semantic(), e.register, e.mask, e.system_value, e.component_type)).collect::<Vec<_>>());
        println!("  OSG1: {:?}", o.iter().map(|e| format!("{}@r{}:{:x} sv{} t{}", e.semantic(), e.register, e.mask, e.system_value, e.component_type)).collect::<Vec<_>>());
        if let Some(parts) = crate::gpu::shader::container_parts(&blob) {
            if let Some(p) = parts.iter().find(|p| p.fourcc == "PSV0") {
                let d = &blob[p.offset..p.offset + p.size];
                println!("  PSV0 {:02x?}", &d[..d.len().min(60)]);
            }
        }
    }
}

/// Geometry mode end to end: an ES program (vertex → LDS, s_setpc to the GS) and a GS
/// program (triangle from LDS through the v0/v1 offsets, a 4-vertex strip = 2 triangles,
/// emit / cut) run as one compute shader; 25 triangles = 2 groups (21 + 4); records checked
/// vertex by vertex (strip winding included), then drawn by the mesh pass-through.
#[test]
fn gpu_geometry_mode_es_gs_strip() {
    if !gpu_ready_wave32() || !mesh_ready() {
        return;
    }
    use super::{dxc, mesh_passthrough, GsInterface, IO_GS_PRIMS, IO_GS_VERTICES, MESH_HEADER};
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap();
    // ES: vertex (v9..v12) → LDS at lane·32, then into the GS
    let mut es = Vec::new();
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 133), ("VSRC1", 20)], None));
    es.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 9), ("DATA1", 10), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    es.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 21), ("DATA0", 11), ("DATA1", 12), ("OFFSET0", 2), ("OFFSET1", 3)], None));
    es.extend(e("S_BARRIER", "ENC_SOPP", &[], None));
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    // GS: EXEC = primitives of this wave (s3[15:8]), vertices from LDS, parallelogram strip
    let mut gs = Vec::new();
    gs.extend(e("S_BFE_U32", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)));
    gs.extend(e("S_BFM_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None));
    gs.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    gs.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 130), ("VSRC1", 0)], None));
    gs.extend(e("V_AND_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 255), ("VSRC1", 30)], Some(0x3FFFC)));
    gs.extend(e("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 144), ("VSRC1", 0)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 31), ("SRC0", 130), ("VSRC1", 31)], None));
    gs.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 32), ("SRC0", 130), ("VSRC1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 40), ("ADDR", 30), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 42), ("ADDR", 31), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    gs.extend(e("DS_READ2_B32", "ENC_DS", &[("VDST", 44), ("ADDR", 32), ("OFFSET0", 0), ("OFFSET1", 1)], None));
    for (d, a, b) in [(46u64, 42u64, 44u64), (47, 43, 45)] {
        gs.extend(e("V_ADD_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None));
        gs.extend(e("V_SUB_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + d), ("VSRC1", d - 6)], None)); // − x0 / − y0
    }
    // output base: s2 + lane · (4 vertices · 32 bytes)
    gs.extend(e("V_MAD_U32_U24", "ENC_VOP3", &[("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(128)));
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 51), ("SRC0", 240)], None)); // 0.5
    gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 52), ("SRC0", 242)], None)); // 1.0
    for (k, (x, y)) in [(40u64, 41u64), (42, 43), (44, 45), (46, 47)].into_iter().enumerate() {
        let o = 8 * k as u64;
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", x), ("DATA1", y), ("OFFSET0", o), ("OFFSET1", o + 1)], None));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 51), ("DATA1", 52), ("OFFSET0", o + 2), ("OFFSET1", o + 3)], None));
        gs.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 53), ("SRC0", 128 + k as u64)], None));
        gs.extend(e("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 50), ("DATA0", 2), ("DATA1", 53), ("OFFSET0", o + 4), ("OFFSET1", o + 5)], None));
        gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x22)], None)); // emit
    }
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x12)], None)); // cut
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None)); // GS done
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));

    let sig = |name: &str, sv: u32, reg: u32, ty: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: ty,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let g = GsInterface {
        es_inputs: vec![sig("POSITION", 0, 0, 3)],
        // two ES output registers: 32-byte ESGS items (the ES above writes at lane · 32)
        es_outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 3)],
        outputs: vec![sig("SV_Position", 1, 0, 3), sig("TEXCOORD", 0, 1, 1)],
        max_vertex_count: 4,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);
    let native = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, geometry_wave32: false, ..Default::default() }).unwrap();
    assert!(native.hlsl.contains("[WaveSize(64)]"));
    assert!(!native.hlsl.contains("xcl_es_phase") && !native.hlsl.contains("GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)"));
    let t = translate(&code, &Options { mode: Mode::Geometry(g.clone()), wave64: true, geometry_wave32: true, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(32)]"));
    assert!(t.hlsl.contains("xcl_es_phase") && t.hlsl.contains("GroupMemoryBarrierWithGroupSync();\n    if (gi < 32u)"));
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let rec = g.record();
    let msb = dxc::compile(&mesh_passthrough(&rec), "main", "ms_6_6", &[]).unwrap();
    let ps_src = "struct V { float4 pos : SV_Position; nointerpolation uint4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return float4(1.0, (float)v.p0.y, 0.0, 1.0); }\n";
    let ps = dxc::compile(ps_src, "main", "ps_6_6", &[]).unwrap();

    // 25 triangles, vertex attributes pre-fetched at 0x1000 (one float4 per vertex)
    let total = 25usize;
    let vtx = |p: usize, i: usize| [p as f32 * 0.03 - 0.6 + i as f32 * 0.2, p as f32 * 0.02 - 0.5 + (i == 2) as u32 as f32 * 0.25, 0.5, 1.0];
    let mut mem = vec![0u32; 0x1000 / 4 + total * 3 * 4];
    for p in 0..total {
        for i in 0..3 {
            let at = 0x1000 / 4 + (p * 3 + i) * 4;
            mem[at..at + 4].copy_from_slice(&vtx(p, i).map(f32::to_bits));
        }
    }
    let mut io = vec![0u32; 128];
    io[IO_GS_PRIMS] = total as u32;
    io[IO_GS_VERTICES] = 0x1000;
    let (px, recs) = runner::mesh_draw(&cs, &msb, &ps, 2, rec.record_dwords(), &io, &mem, 32, [0.0; 4]).unwrap();
    let (rd, vd) = (rec.record_dwords(), rec.vert_dwords());
    for grp in 0..2usize {
        let r = &recs[grp * rd..(grp + 1) * rd];
        let np = if grp == 0 { 21 } else { 4 };
        assert_eq!((r[0], r[1]), (4 * np as u32, 2 * np as u32), "group {grp}: vertices / triangles");
        for lane in 0..np {
            let p = grp * 21 + lane;
            let (a, b, c) = (vtx(p, 0), vtx(p, 1), vtx(p, 2));
            let xy = [[a[0], a[1]], [b[0], b[1]], [c[0], c[1]], [b[0] + c[0] - a[0], b[1] + c[1] - a[1]]];
            for (k, want) in xy.iter().enumerate() {
                let v = MESH_HEADER + (4 * lane + k) * vd;
                let pos = [f32::from_bits(r[v]), f32::from_bits(r[v + 1]), f32::from_bits(r[v + 2]), f32::from_bits(r[v + 3])];
                assert_eq!(pos, [want[0], want[1], 0.5, 1.0], "group {grp} lane {lane} vertex {k}");
                assert_eq!((r[v + 4], r[v + 5]), (p as u32, k as u32), "group {grp} lane {lane} vertex {k}: TEXCOORD0");
            }
            let pk = |i: usize, j: usize, l: usize| (i | j << 10 | l << 20) as u32;
            let tris = MESH_HEADER + rec.max_vertices as usize * vd;
            let b0 = 4 * lane;
            assert_eq!(r[tris + 2 * lane], pk(b0, b0 + 1, b0 + 2), "group {grp} lane {lane}: strip triangle 0");
            assert_eq!(r[tris + 2 * lane + 1], pk(b0 + 2, b0 + 1, b0 + 3), "group {grp} lane {lane}: strip triangle 1 (odd: swapped)");
        }
    }
    let covered = px.iter().filter(|p| p[0] == 1.0).count();
    assert!(covered > 20, "pass-through drew the GS output: {covered} pixels");
}

/// v_mbcnt over a v_cmp mask in a geometry program (the compaction index of an ES / GS):
/// lanes 32..63 count the active lanes of 0..31 too. The split wave32 form only has its own
/// half of a mask, so it must refuse the program; whatever translates must give the logical
/// wave64 counts (the compute pass alone: no mesh shaders needed).
#[test]
fn gpu_geometry_wave32_prefix_count_over_a_mask() {
    if !gpu_ready_wave32() {
        return;
    }
    use super::{dxc, translate_geometry, GsInterface, IO_GS_PRIMS};
    let e = |n: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>| encode(n, enc, f, lit).unwrap();
    let mut es = Vec::new();
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None)); // lane
    es.extend(e("V_AND_B32", "ENC_VOP2", &[("VDST", 21), ("SRC0", 131), ("VSRC1", 20)], None));
    es.extend(e("V_CMP_NE_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 21)], None)); // vcc = lane & 3 != 0
    es.extend(e("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 10), ("SRC0", 168), ("SRC1", V + 20)], None)); // s[10:11] = lane < 40
    // v22: below this lane in VCC, v23: in s[10:11], v24: in EXEC (= VCC; the others keep 0xFFFF)
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", 106), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 22), ("SRC0", 107), ("SRC1", V + 22)], None));
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 10), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 11), ("SRC1", V + 23)], None));
    es.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", 255)], Some(0xFFFF)));
    es.extend(e("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 106)], None));
    es.extend(e("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 126), ("SRC1", 128)], None));
    es.extend(e("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 24), ("SRC0", 127), ("SRC1", V + 24)], None));
    es.extend(e("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 12)], None));
    // s[8:9] + lane · 4: v22 at +0, v23 at +256, v24 at +512
    es.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 25), ("SRC0", 130), ("VSRC1", 20)], None));
    for (k, d) in [22u64, 23, 24].into_iter().enumerate() {
        es.extend(e("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 25), ("DATA", d), ("SADDR", 8), ("OFFSET", 256 * k as u64)], None));
    }
    es.extend(e("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None));
    let mut gs = Vec::new();
    gs.extend(e("S_SENDMSG", "ENC_SOPP", &[("SIMM16", 0x3)], None)); // GS done, nothing emitted
    gs.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: reg,
        mask: 0xF,
        rw_mask: 0xF,
        stream: 0,
    };
    let g = GsInterface {
        es_inputs: vec![],
        es_outputs: vec![sig("SV_Position", 1, 0)],
        outputs: vec![sig("SV_Position", 1, 0)],
        max_vertex_count: 3,
        gs_start: Some(es.len() as u64),
    };
    let mut code = es;
    code.extend_from_slice(&gs);

    // what one wave64 computes
    let below = |m: u64, l: usize| (m & ((1u64 << l) - 1)).count_ones();
    let (vcc, s10) = ((0..64).filter(|l| l & 3 != 0).fold(0u64, |m, l| m | 1 << l), (1u64 << 40) - 1);
    let want: Vec<[u32; 3]> = (0..64).map(|l| [below(vcc, l), below(s10, l), if vcc >> l & 1 == 1 { below(vcc, l) } else { 0xFFFF }]).collect();
    let mut io = vec![0u32; 128];
    io[8] = 0x400;
    io[IO_GS_PRIMS] = 21;
    let run = |t: &super::Translation| {
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let cs = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
        let mem = runner::dispatch(&cs, [1, 1, 1], &io, &vec![0u32; 0x400 / 4 + 3 * 64]).unwrap();
        let got: Vec<[u32; 3]> = (0..64).map(|l| [0, 1, 2].map(|k| mem[0x100 + 64 * k + l])).collect();
        let wrong: Vec<_> = (0..64).filter(|&l| got[l] != want[l]).map(|l| (l, got[l], want[l])).collect();
        assert!(wrong.is_empty(), "lanes (got [vcc, s[10:11], exec], want): {wrong:?}");
    };
    match translate_geometry(&code, g.clone(), true) {
        Ok(t) => run(&t),
        Err(err) => assert!(err.contains("prefix count over a whole-wave mask"), "{err}"),
    }
    // the native wave64 form keeps the program, and computes the reference where it can run
    let native = translate_geometry(&code, g, false).unwrap();
    if runner::wave64_supported() == Ok(true) {
        run(&native);
    } else {
        runner::skip_test("no wave64");
    }
}

/// A DS read whose destination overlaps its address register reads the address first
/// (retail GS code: `ds_read2_b32 v[0:1], v0 offset1:1`).
#[test]
fn gpu_ds_read_destination_overlaps_address() {
    if !gpu_ready_wave32() {
        return;
    }
    let mut code = Vec::new();
    code.extend(encode("DS_WRITE2_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", 2), ("DATA1", 3), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("DS_READ2_B32", "ENC_DS", &[("VDST", 1), ("ADDR", 1), ("OFFSET0", 0), ("OFFSET1", 1)], None).unwrap());
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[1] = 8 * l as u32; // address
        r[2] = 1000 + l as u32;
        r[3] = 2000 + l as u32;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (1000 + l as u32, 2000 + l as u32), "lane {l}");
    }
}

/// Ray tracing: image_bvh(64)_intersect_ray on a hand-built BVH (box32 root with 4 children,
/// a box16 node, triangles), several rays; sorted and unsorted box results, triangle hit as
/// (t, denominator, i, j) numerators, misses, and the 64-bit node pointer form.
#[test]
fn gpu_bvh_intersect_ray() {
    if !gpu_ready_wave32() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let (root, box16, tri_a, tri_b) = (5u32, (0x80u32 >> 3) | 4, 0xC0u32 >> 3, 0x100u32 >> 3);
    // root box32 at 0: children + boxes (min xyz, max xyz)
    w(&mut mem, 0, &[tri_a, box16, 0xFFFF_FFFF, tri_b]);
    w(&mut mem, 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0])); // triangle A's box: entry t = 5
    w(&mut mem, 40, &ff(&[-1.0, -1.0, 2.0, 1.0, 1.0, 3.0])); // box16 node's box: t = 2
    w(&mut mem, 64, &ff(&[-1.0, -1.0, 0.5, 1.0, 1.0, 0.7])); // invalid child: never reported
    w(&mut mem, 88, &ff(&[10.0, -1.0, 8.0, 11.0, 1.0, 9.0])); // off the ray
    // box16 node at 0x80: one child (triangle B) in z 6..7, half floats
    let h = |x: f32| half_bits(x);
    w(&mut mem, 0x80, &[tri_b, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF]);
    w(&mut mem, 0x90, &[h(-1.0) | h(-1.0) << 16, h(6.0) | h(1.0) << 16, h(1.0) | h(7.0) << 16]);
    // triangles: A in z = 5 (hit at u = v = 0.5 from the origin), B in z = 6.5
    w(&mut mem, 0xC0, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    w(&mut mem, 0x100, &ff(&[-1.0, -1.0, 6.5, 1.0, -1.0, 6.5, -1.0, 1.0, 6.5]));
    // triangle id 9: I = vertex 1, J = vertex 2 (no rotation)
    w(&mut mem, 0xC0 + 60, &[9]);
    w(&mut mem, 0x100 + 60, &[9]);
    let mimg = |n: &'static str, vdata: u64, vaddr: u64, srsrc: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 100, 0, 2)); // root, sorting
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 104, 0, 3)); // root, no sorting
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 20, 2)); // box16 node
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 112, 40, 2)); // triangle A
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 116, 60, 2)); // triangle A, 64-bit node id
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    st.sgpr[12..16].copy_from_slice(&[(base >> 8) as u32, 0, u32::MAX, 0]);
    // lane 0: ray along +z from the origin; lane 1: the same with t max 3.5; lane 2: from (5, 5)
    let ray = |l: usize| -> ([f32; 3], f32) {
        match l % 3 {
            0 => ([0.0, 0.0, 0.0], 100.0),
            1 => ([0.0, 0.0, 0.0], 3.5),
            _ => ([5.0, 5.0, 0.0], 100.0),
        }
    };
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let (o, tmax) = ray(l);
        let rest = |r: &mut Vec<u32>, at: usize| {
            let v = [tmax, o[0], o[1], o[2], 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
            for (k, x) in v.iter().enumerate() {
                r[at + k] = x.to_bits();
            }
        };
        r[0] = root;
        rest(r, 1);
        r[20] = box16;
        rest(r, 21);
        r[40] = tri_a;
        rest(r, 41);
        r[60] = tri_a;
        r[61] = 0;
        rest(r, 62);
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    const F: u32 = 0xFFFF_FFFF;
    for l in 0..64usize {
        let v = &res.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        match l % 3 {
            0 => {
                assert_eq!(q(100), [box16, tri_a, F, F], "lane {l}: box32, closest first");
                assert_eq!(q(104), [tri_a, box16, F, F], "lane {l}: box32 without sorting keeps child order");
                assert_eq!(q(108), [tri_b, F, F, F], "lane {l}: box16");
            }
            1 => {
                assert_eq!(q(100), [box16, F, F, F], "lane {l}: t max 3.5 excludes the box at t = 5");
                assert_eq!(q(108), [F, F, F, F], "lane {l}: box16 at t = 6 beyond t max");
            }
            _ => {
                assert_eq!(q(100), [F, F, F, F], "lane {l}: ray from (5, 5) misses every box");
                assert_eq!(q(112)[0], f32::INFINITY.to_bits(), "lane {l}: triangle missed → t = inf");
            }
        }
        if l % 3 != 2 {
            let tri = q(112).map(f32::from_bits);
            let (t, i, j) = (tri[0] / tri[1], tri[2] / tri[1], tri[3] / tri[1]);
            assert_eq!((t, i, j), (5.0, 0.5, 0.5), "lane {l}: triangle hit (t, i, j)");
            assert_eq!(q(116), q(112), "lane {l}: 64-bit node pointer gives the same");
        }
    }
}

/// Checks the compute ABI read from the FSSM stage descriptor (+0x38 COMPUTE_PGM_RSRC2,
/// +0x3C.. numthreads) on every CS of the pack against PSV0 and the SGPRs/VGPRs the code reads.
#[test]
#[ignore]
fn compute_abi_survey() {
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let data = std::fs::read(path).unwrap();
    let (mut n, mut threads_ok, mut sgpr_ok, mut vgpr_ok) = (0, 0, 0, 0);
    for r in crate::gpu::pack::scan(&data, 0x40) {
        // compute records are recognised by their hardware stage below, not by name
        let Some(blob) = crate::gpu::pack::inflate(&data, &r) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        let Some(f) = sh.fssm.iter().find(|f| f.stages.first().is_some_and(|s| s.hw == 6)) else { continue };
        let d = f.stages[0].offset;
        let rd = |o: usize| u32::from_le_bytes(f.data[d + o..d + o + 4].try_into().unwrap());
        let rsrc2 = rd(0x38);
        let nt = [rd(0x3C), rd(0x40), rd(0x44)];
        let user = (rsrc2 >> 1) & 31;
        let tg = [(rsrc2 >> 7) & 1, (rsrc2 >> 8) & 1, (rsrc2 >> 9) & 1, (rsrc2 >> 10) & 1];
        let tid = (rsrc2 >> 11) & 3;
        let sys = tg.iter().sum::<u32>();
        n += 1;
        let psv = crate::gpu::dxbc::psv(&blob).and_then(|p| p.numthreads);
        threads_ok += (psv == Some(nt)) as usize;
        // registers read before written
        let p = f.programs[0];
        let code = &f.data[p.offset..p.offset + p.len];
        let (mut sr, mut sw, mut vr, mut vw) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        let mut off = 0;
        while off + 4 <= code.len() {
            let Ok(dd) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { break };
            off += dd.len;
            let (r, w) = crate::gpu::rdna2::sgpr_uses(&dd);
            sr.extend(r.into_iter().filter(|x| !sw.contains(x) && *x < 106));
            sw.extend(w);
            let (r, w) = crate::gpu::rdna2::vgpr_uses(&dd);
            vr.extend(r.into_iter().filter(|x| !vw.contains(x)));
            vw.extend(w);
        }
        let max_s = sr.iter().max().copied().unwrap_or(0);
        let s_in = max_s < (user + sys) as u64;
        sgpr_ok += s_in as usize;
        let v_in = vr.iter().all(|&v| v <= tid as u64);
        vgpr_ok += v_in as usize;
        if !(s_in && v_in && psv == Some(nt)) {
            println!("{}: rsrc2 {rsrc2:#x} user {user} tgid {tg:?} tid {tid} threads {nt:?} psv {psv:?} SGPRs read {sr:?} VGPR {vr:?}", r.name);
        }
    }
    println!("CS: {n}, numthreads = PSV0: {threads_ok}, SGPRs within user+system: {sgpr_ok}, VGPRs within the thread ID: {vgpr_ok}");
}

/// Compute mode with the console dispatch ABI: numthreads 16×8 (two wave64 waves per group),
/// 2×3 groups; group ids from the SGPRs after the user data, thread ids in v0/v1; LDS shared
/// across the two waves through a barrier.
#[test]
fn gpu_compute_mode_dispatch_abi() {
    if !gpu_ready() {
        return;
    }
    let t = translate(&dispatch_abi_program(), &Options { mode: Mode::Compute(dispatch_abi_cs()), wave64: true, ..Default::default() }).unwrap();
    check_dispatch_abi(&t);
}

fn dispatch_abi_cs() -> super::CsInterface {
    super::CsInterface { threads: [16, 8, 1], user_sgprs: 2, tgid: [true, true, false], tg_size: false, tid_components: 2 }
}

/// 128 threads (two waves): each writes its id, and through LDS + s_barrier the id of the
/// thread at the mirrored index in the other wave.
fn dispatch_abi_program() -> Vec<u8> {
    prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 132), ("VSRC1", 1)], None), // y · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 2), ("VSRC1", 0)], None), // index in group
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 2), ("SSRC1", 132)], None), // group x · 16
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 4), ("VSRC1", 0)], None), // global x
        ("S_LSHL_B32", "ENC_SOP2", &[("SDST", 5), ("SSRC0", 3), ("SSRC1", 131)], None), // group y · 8
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 5), ("VSRC1", 1)], None), // global y
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 144), ("VSRC1", 4)], None),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 5), ("SRC0", V + 5), ("VSRC1", 3)], None), // x | y << 16
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 2)], None),
        ("DS_WRITE_B32", "ENC_DS", &[("ADDR", 6), ("DATA0", 5)], None),
        ("S_BARRIER", "ENC_SOPP", &[], None),
        ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 255), ("VSRC1", 2)], Some(127)), // partner in the other wave
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 130), ("VSRC1", 7)], None),
        ("DS_READ_B32", "ENC_DS", &[("VDST", 12), ("ADDR", 7)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 5)], None),
        ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 4), ("SRC1", 1)], None), // y · width
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", V + 9), ("VSRC1", 3)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 131), ("VSRC1", 9)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 9), ("SRC0", 0), ("VSRC1", 9)], None), // + output base
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX2", "ENC_FLAT_GLBL", &[("ADDR", 9), ("DATA", 11), ("SADDR", 0x7D)], None),
    ])
}

fn check_dispatch_abi(t: &super::Translation) {
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let (w, h, base) = (32usize, 24usize, 0x100usize);
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    io[1] = w as u32;
    let out = runner::dispatch(&dxil, [2, 3, 1], &io, &vec![0u32; (base + w * h * 8) / 4]).unwrap();
    for y in 0..h {
        for x in 0..w {
            let at = (base + (y * w + x) * 8) / 4;
            assert_eq!(out[at], (x | y << 16) as u32, "({x}, {y}): own value");
            let (gx, gy, i) = (x / 16, y / 8, (y % 8) * 16 + x % 16);
            let (px, py) = (gx * 16 + (127 - i) % 16, gy * 8 + (127 - i) / 16);
            assert_eq!(out[at + 1], (px | py << 16) as u32, "({x}, {y}): value of thread {} through LDS", 127 - i);
        }
    }
}

/// The same wave64 compute program on an adapter without 64-lane waves (NVIDIA): each wave
/// runs as two native wave32 halves, and the result is the same.
#[test]
fn gpu_compute_wave64_as_wave32_halves() {
    if !gpu_ready_wave32() {
        return;
    }
    let opt = Options { mode: Mode::Compute(dispatch_abi_cs()), wave64: true, compute_wave32: true, ..Default::default() };
    let t = super::translate_compute(&dispatch_abi_program(), opt).unwrap();
    assert!(t.hlsl.contains("[WaveSize(32)]") && t.hlsl.contains("xcl_logical_wave64 = true"), "{}", t.hlsl);
    check_dispatch_abi(&t);
}

/// EXEC per half in a split wave64 compute program: v_cmpx keeps lanes 0..39, so the high
/// half (lanes 32..63) keeps 8 lanes, which it reads from its own word of the mask.
#[test]
fn gpu_compute_wave32_halves_keep_their_exec_bits() {
    if !gpu_ready_wave32() {
        return;
    }
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let p = prog(&[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None), // v0 · 4
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)], None),     // + output base (s0)
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 128)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 129), ("VSRC1", 0)], None), // v0 + 1
        ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 0)], Some(40)),         // exec = v0 < 40
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 3), ("SADDR", 0x7D)], None),
    ]);
    let opt = Options { mode: Mode::Compute(cs), wave64: true, compute_wave32: true, ..Default::default() };
    let t = super::translate_compute(&p, opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
    let base = 0x100usize;
    let mut io = vec![0u32; 128];
    io[0] = base as u32;
    let out = runner::dispatch(&dxil, [1, 1, 1], &io, &vec![0u32; (base + 64 * 4) / 4]).unwrap();
    for l in 0..64 {
        assert_eq!(out[base / 4 + l], if l < 40 { l as u32 + 1 } else { 0 }, "lane {l}");
    }
}

/// A split wave64 compute program that needs the mask of the whole wave is refused, as split
/// geometry and mesh programs are; with 64-lane waves it translates.
#[test]
fn split_compute_refuses_cross_half_semantics() {
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 0, tgid: [false; 3], tg_size: false, tid_components: 1 };
    let p = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 138), ("VSRC1", 0)], None),
        ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None), // lanes in vcc
    ]);
    let opt = Options { mode: Mode::Compute(cs), wave64: true, ..Default::default() };
    assert!(super::translate_compute(&p, opt.clone()).is_ok());
    let e = super::translate_compute(&p, Options { compute_wave32: true, ..opt }).err().expect("split compute must refuse");
    assert!(e.contains("compute program requires unsupported cross-half wave64 semantics") && e.contains("whole-wave mask reduction"), "{e}");
}

/// A wave32 program in a 64-lane hardware wave (what VS / PS may get from the driver): the
/// two halves behave as separate waves – ballots, lane counts, readlane and mbcnt per half.
#[test]
fn gpu_wave32_program_in_wave64_hardware() {
    if !gpu_ready() {
        return;
    }
    let p = prog(&[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 138), ("VSRC1", 0)], None), // vcc_lo = lanes with v0 < 10
        ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 0)], None),
        ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", 131)], None), // s1 = v0 of lane 3
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 1)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 193), ("SRC1", 128)], None),
    ]);
    let t = translate(&p, &Options { mode: Mode::Kernel, wave64: false, hw_wave: 64, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("[WaveSize(64)]") && t.hlsl.contains("xcl_half"), "{}", t.hlsl);
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    let mut st = KernelState::default();
    st.sgpr[126] = u32::MAX; // exec of each half
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = (l % 32 + if l >= 32 { 5 } else { 0 }) as u32; // second half shifted by 5
    }
    let r = runner::run(&dxil, &st, &[]).unwrap();
    for l in 0..64 {
        let half = l / 32;
        let v = &r.vgpr[l];
        assert_eq!(v[1], if half == 0 { 10 } else { 5 }, "lane {l}: active lanes of its own half");
        assert_eq!(v[2], 3 + 5 * half as u32, "lane {l}: readlane 3 of its own half");
        assert_eq!(v[3], (l % 32) as u32, "lane {l}: lane index within the half");
    }
}

/// SV_VertexID / SV_InstanceID listed in ISG1 with a register arrive like attributes,
/// v[4 + 4r] (retail VInstanced* shaders read the instance id in v8 = register 1).
#[test]
fn vs_system_values_in_isg1_registers() {
    use super::VsInterface;
    let sig = |name: &str, sv: u32, reg: u32| crate::gpu::dxbc::SigElement {
        name: name.into(),
        index: 0,
        system_value: sv,
        component_type: if sv == 0 { 3 } else { 1 },
        register: reg,
        mask: if sv == 0 { 7 } else { 1 },
        rw_mask: 1,
        stream: 0,
    };
    let vs = VsInterface {
        inputs: vec![sig("POSITION", 0, 0), sig("SV_InstanceID", 8, 1), sig("SV_VertexID", 6, 2)],
        outputs: vec![sig("SV_Position", 1, 0)],
        attr_base: 4,
        params: None,
    };
    let code = prog(&[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 8), ("VSRC1", 12)], None),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 21), ("SRC0", V + 4), ("VSRC1", 5)], None),
    ]);
    let t = translate(&code, &Options { mode: Mode::Vertex(vs), wave64: true, ..Default::default() }).unwrap();
    assert!(t.hlsl.contains("uint v8 = xi.xcl_iid;") && t.hlsl.contains("uint v12 = xi.xcl_vid;"), "{}", t.hlsl);
    assert!(t.hlsl.contains("uint v4 = asuint(xi.a0[0]);"), "{}", t.hlsl);
}

#[test]
#[ignore]
fn probe_standard_swizzle_support() {
    use windows::Win32::Graphics::Direct3D12::*;
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0, &mut dev).unwrap();
        let dev = dev.unwrap();
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut _, std::mem::size_of_val(&o) as u32).unwrap();
        println!("StandardSwizzle64KBSupported = {:?}", o.StandardSwizzle64KBSupported);
    }
}

/// A BVH from the builder (gpu::bvh) through the translated image_bvh_intersect_ray: every
/// lane tests one built node (box32 or triangle) with its own ray; the GPU must agree with
/// the CPU node test the builder's traversal uses.
#[test]
fn gpu_built_bvh_nodes_match_the_cpu_node_test() {
    use crate::gpu::bvh;
    if !gpu_ready() {
        return;
    }
    let tris = bvh::tests::random_tris(40, 5);
    let b = bvh::build(&tris, 0);
    // every node id, root first
    let mut nodes = vec![];
    let mut todo = vec![b.root];
    while let Some(n) = todo.pop() {
        nodes.push(n);
        if n & 7 == bvh::BOX32 {
            let at = ((n & !7) as usize) << 3;
            for k in 0..4 {
                let c = u32::from_le_bytes(b.data[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
                if c != bvh::INVALID {
                    todo.push(c);
                }
            }
        }
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; (base + b.data.len()) / 4 + 64];
    for (k, c) in b.data.chunks(4).enumerate() {
        mem[base / 4 + k] = u32::from_le_bytes(c.try_into().unwrap());
    }
    let mut code = encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut r = bvh::tests::rng(17);
    let mut rays = vec![];
    for (l, v) in st.vgpr.iter_mut().enumerate() {
        let node = nodes[l % nodes.len()];
        let o = [r(-12.0, 12.0), r(-12.0, 12.0), r(-12.0, 12.0)];
        let to = [r(-6.0, 6.0), r(-6.0, 6.0), r(-6.0, 6.0)];
        let d = [to[0] - o[0], to[1] - o[1], to[2] - o[2]];
        let inv = d.map(|x| 1.0 / x);
        v[0] = node;
        for (k, x) in [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]].iter().enumerate() {
            v[1 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    let (mut boxes, mut tri_hits) = (0, 0);
    for (l, &(node, o, d, inv)) in rays.iter().enumerate() {
        let gpu: [u32; 4] = res.vgpr[l][100..104].try_into().unwrap();
        let cpu = bvh::intersect_node(&b.data, node, 100.0, o, d, inv, true);
        if node & 7 == bvh::BOX32 {
            assert_eq!(gpu, cpu, "lane {l}: box node {node:#x}");
            boxes += 1;
        } else {
            let (g, c) = (gpu.map(f32::from_bits), cpu.map(f32::from_bits));
            assert_eq!(g[0].is_infinite(), c[0].is_infinite(), "lane {l}: triangle {node:#x} hit / miss");
            if !c[0].is_infinite() {
                assert!((g[0] / g[1] - c[0] / c[1]).abs() < 1e-4, "lane {l}: t {g:?} vs {c:?}");
                tri_hits += 1;
            }
        }
    }
    assert!(boxes > 10 && tri_hits > 0, "boxes {boxes}, triangle hits {tri_hits}");
}

/// The console's node forms (seen in the console builder's output): a box16 node and a GPURT
/// triangle pair – type 0 = (v0, v1, v2), type 1 = (v1, v3, v2), barycentrics put back in the
/// original vertex order by the triangle id. GPU (HLSL emulation) = CPU mirror, and the
/// barycentrics of a known point come out right.
#[test]
fn gpu_triangle_pair_and_box16_nodes() {
    use crate::gpu::bvh;
    if !gpu_ready() {
        return;
    }
    let base = 0x1000usize;
    let mut mem = vec![0u32; 0x2000 / 4];
    let w = |mem: &mut Vec<u32>, at: usize, vals: &[u32]| mem[(base + at) / 4..(base + at) / 4 + vals.len()].copy_from_slice(vals);
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // corners of a quad in z = 5; triangle 0 = (p0, p1, p2) stored rotated: v0 = p1, v1 = p2,
    // v2 = p0, so its I barycentric (weight of p1) is vertex 0 and J (p2) vertex 1
    let (p0, p1, p2, p3) = ([-1.0f32, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0]);
    let pair = 0x40usize;
    w(&mut mem, pair, &ff(&[p1, p2, p0, p3].concat()));
    // triangle 1 = (v1, v3, v2) = (p2, p3, p0), original order (p0, p3, p2): I (p3) is vertex 1
    // of the triple, J (p2) vertex 0. Triangle id: byte 0 = 0x04, byte 1 = 0x01
    w(&mut mem, pair + 48, &[0, 11, 12, 0x0104]);
    // box16 root with the two triangles
    let h = |x: f32| half_bits(x);
    let (t0, t1) = ((pair as u32) >> 3, (pair as u32) >> 3 | 1);
    w(&mut mem, 0, &[t0, t1, 0xFFFF_FFFF, 0xFFFF_FFFF]);
    for k in 0..2 {
        w(&mut mem, 16 + 12 * k, &[h(-1.0) | h(-1.0) << 16, h(4.5) | h(1.0) << 16, h(1.0) | h(5.5) << 16]);
    }
    let root = 4u32; // box16 at 0
    let mut code = encode("IMAGE_BVH_INTERSECT_RAY", "ENC_MIMG", &[("VDATA", 100), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15)], None).unwrap();
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&[(base >> 8) as u32, 1 << 31, u32::MAX, 0]);
    let mut r = bvh::tests::rng(5);
    let nodes = [root, t0, t1];
    let mut rays = vec![];
    for (l, v) in st.vgpr.iter_mut().enumerate() {
        let node = nodes[l % 3];
        let o = [r(-1.5, 1.5), r(-1.5, 1.5), 0.0];
        let d = [r(-0.1, 0.1), r(-0.1, 0.1), 1.0];
        let inv = d.map(|x| 1.0 / x);
        v[0] = node;
        for (k, x) in [100.0f32, o[0], o[1], o[2], d[0], d[1], d[2], inv[0], inv[1], inv[2]].iter().enumerate() {
            v[1 + k] = x.to_bits();
        }
        rays.push((node, o, d, inv));
    }
    let res = runner::run_code(&code, &st, &mem).unwrap();
    let data: Vec<u8> = mem[base / 4..].iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut checked = 0;
    for (l, &(node, o, d, inv)) in rays.iter().enumerate() {
        let gpu: [u32; 4] = res.vgpr[l][100..104].try_into().unwrap();
        let cpu = bvh::intersect_node(&data, node, 100.0, o, d, inv, true);
        if node == root {
            assert_eq!(gpu, cpu, "lane {l}: box16");
            continue;
        }
        let (g, c) = (gpu.map(f32::from_bits), cpu.map(f32::from_bits));
        assert_eq!(g[0].is_infinite(), c[0].is_infinite(), "lane {l}: hit / miss");
        if c[0].is_infinite() {
            continue;
        }
        for k in 0..4 {
            assert!((g[k] / g[1] - c[k] / c[1]).abs() < 1e-5, "lane {l}: {g:?} vs {c:?}");
        }
        // hit point from the barycentrics in the ORIGINAL order of the stored triangle
        let (t, bi, bj) = (c[0] / c[1], c[2] / c[1], c[3] / c[1]);
        let hit = [0, 1, 2].map(|a| o[a] + t * d[a]);
        if node == t0 {
            let q = [0, 1, 2].map(|a| (1.0 - bi - bj) * p0[a] + bi * p1[a] + bj * p2[a]);
            assert!((0..3).all(|a| (q[a] - hit[a]).abs() < 1e-4), "lane {l}: triangle 0 barycentrics {q:?} vs {hit:?}");
        } else {
            let q = [0, 1, 2].map(|a| (1.0 - bi - bj) * p0[a] + bi * p3[a] + bj * p2[a]);
            assert!((0..3).all(|a| (q[a] - hit[a]).abs() < 1e-4), "lane {l}: triangle 1 barycentrics {q:?} vs {hit:?}");
        }
        checked += 1;
    }
    assert!(checked > 5, "{checked} triangle hits");
}

/// Multi-window `image_bvh64_intersect_ray` (full 64-bit node pointer into UMA window 1 with
/// base = 0, plus 48-bit base in window 1) and axis-aligned grazing ray / NaN-disabled box.
#[test]
fn gpu_bvh64_intersect_ray_across_memory_windows_and_grazing() {
    if !gpu_ready() {
        return;
    }
    let arena = crate::gpu::uma::Uma::with_windows(2, 1 << 20).unwrap();
    let (w0, w1) = (arena.window(0).0 as u64, arena.window(1).0 as u64);
    assert_eq!(w1 - w0, 1 << 32);
    let write_words = |addr: u64, vals: &[u32]| unsafe {
        let p = addr as *mut u32;
        for (k, &v) in vals.iter().enumerate() {
            *p.add(k) = v;
        }
    };
    let ff = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let base_off = 0x1000u64;
    let bvh_addr = w1 + base_off;
    let tri_rel = 0x80u32 >> 3;
    let nan_child = 0xC0u32 >> 3;
    let inv_child = 0x100u32 >> 3;
    // Window 1 root box32 at bvh_addr: child 0 = normal box [-1..1, -1..1, 5..6],
    // child 1 = NaN-disabled box, child 2 = inverted box (mn > mx), child 3 = invalid
    write_words(bvh_addr, &[tri_rel, nan_child, inv_child, u32::MAX]);
    write_words(bvh_addr + 16, &ff(&[-1.0, -1.0, 5.0, 1.0, 1.0, 6.0]));
    write_words(bvh_addr + 40, &ff(&[f32::NAN, -1.0, 2.0, 1.0, 1.0, 3.0]));
    write_words(bvh_addr + 64, &ff(&[2.0, -1.0, 2.0, -2.0, 1.0, 3.0]));
    // Triangle at bvh_addr + 0x80 in z = 5
    write_words(bvh_addr + 0x80, &ff(&[-1.0, -1.0, 5.0, 1.0, -1.0, 5.0, -1.0, 1.0, 5.0]));
    write_words(bvh_addr + 0x80 + 60, &[9]);
    // Poison window 0 at the same low offset: reading window 0 would return bogus child 0xDEAD_BEEF
    write_words(w0 + base_off, &[0xDEAD_BEEF, u32::MAX, u32::MAX, u32::MAX]);
    write_words(w0 + base_off + 16, &ff(&[-100.0, -100.0, 1.0, 100.0, 100.0, 2.0]));

    let mimg = |n: &'static str, vdata: u64, vaddr: u64, srsrc: u64| encode(n, "ENC_MIMG", &[("VDATA", vdata), ("VADDR", vaddr), ("SRSRC", srsrc), ("DMASK", 15)], None).unwrap();
    let mut code = Vec::new();
    // s[8:11] = base 0 (box sort enabled), v[0:1] = full 64-bit node pointer into window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 100, 0, 2));
    // s[8:11] = base 0, v[20:21] = full 64-bit triangle node pointer in window 1
    code.extend(mimg("IMAGE_BVH64_INTERSECT_RAY", 104, 20, 2));
    // s[12:15] = 48-bit base pointing at window 1 (`bvh_words(bvh_addr, true)`), v40 = 32-bit root
    code.extend(mimg("IMAGE_BVH_INTERSECT_RAY", 108, 40, 3));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());

    let t = translate(&code, &Options { mem_windows: 2, ..Default::default() }).unwrap();
    let dxil = super::dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).map_err(|e| format!("{e}\n{}", t.hlsl)).unwrap();

    let mut st = KernelState::default();
    st.sgpr[8..12].copy_from_slice(&crate::gpu::desc::bvh_words(0, true));
    st.sgpr[12..16].copy_from_slice(&crate::gpu::desc::bvh_words(bvh_addr, true));
    let root64 = (bvh_addr >> 3) | 5;
    let tri64 = (bvh_addr + 0x80) >> 3;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        // Even lanes graze the box edge (o.x = -1.0 == mn.x, dx = 0, inv.x = +inf);
        // odd lanes fire through the center (0, 0, 0) along +z.
        let ox = if l % 2 == 0 { -1.0f32 } else { 0.0f32 };
        let ray = [100.0f32, ox, 0.0, 0.0, 0.0, 0.0, 1.0, f32::INFINITY, f32::INFINITY, 1.0];
        let put_ray = |r: &mut Vec<u32>, at: usize| {
            for (k, x) in ray.iter().enumerate() {
                r[at + k] = x.to_bits();
            }
        };
        r[0] = root64 as u32;
        r[1] = (root64 >> 32) as u32;
        put_ray(r, 2);
        r[20] = tri64 as u32;
        r[21] = (tri64 >> 32) as u32;
        put_ray(r, 22);
        r[40] = 5; // relative root box32 at bvh_addr
        put_ray(r, 41);
    }
    let res = runner::run_in_arena(&dxil, &st, &arena).unwrap();
    const F: u32 = u32::MAX;
    for l in 0..64usize {
        let v = &res.vgpr[l];
        let q = |at: usize| [v[at], v[at + 1], v[at + 2], v[at + 3]];
        assert_eq!(q(100), [tri_rel, F, F, F], "lane {l}: 64-bit box32 in window 1 (grazing/NaN/inverted)");
        assert_eq!(q(108), [tri_rel, F, F, F], "lane {l}: 48-bit base box32 in window 1");
        let tri = q(104).map(f32::from_bits);
        assert!(!tri[0].is_infinite() && ((tri[0] / tri[1]) - 5.0).abs() < 1e-4, "lane {l}: 64-bit triangle in window 1 → {tri:?}");
    }
}

/// Instructions first met in XboxBvhBuilder.dll's shaders: global atomics (float min / max by
/// a CAS loop), 8 / 16-bit buffer loads / stores and 8 / 16-bit LDS accesses – neighbouring
/// lanes share dwords, so a non-atomic read-modify-write would lose writes.
#[test]
fn gpu_global_atomics_and_sub_dword_access() {
    if !gpu_ready_wave32() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mubuf = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_MUBUF", f, None).unwrap();
    let ds = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_DS", f, None).unwrap();
    let mut code = Vec::new();
    // global atomics on s[2:3] + v30 (= 0 for every lane) + offset
    code.extend(glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 40), ("ADDR", 30), ("DATA", 31), ("SADDR", 2), ("OFFSET", 0), ("GLC", 1)]));
    code.extend(glbl("GLOBAL_ATOMIC_SUB", &[("VDST", 41), ("ADDR", 30), ("DATA", 35), ("SADDR", 2), ("OFFSET", 4)]));
    code.extend(glbl("GLOBAL_ATOMIC_FMAX", &[("VDST", 42), ("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_ATOMIC_FMIN", &[("VDST", 43), ("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", 12)]));
    code.extend(glbl("GLOBAL_ATOMIC_SMAX", &[("VDST", 44), ("ADDR", 30), ("DATA", 33), ("SADDR", 2), ("OFFSET", 16)]));
    // V# s[24:27]: raw at 0x700; short at lane*2, byte at 0x80 + lane (offen)
    code.extend(mubuf("BUFFER_STORE_SHORT", &[("VDATA", 34), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_STORE_BYTE", &[("VDATA", 36), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_USHORT", &[("VDATA", 45), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_SSHORT", &[("VDATA", 46), ("VADDR", 2), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_UBYTE", &[("VDATA", 47), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(mubuf("BUFFER_LOAD_SBYTE", &[("VDATA", 48), ("VADDR", 3), ("SRSRC", 6), ("SOFFSET", 128), ("OFFEN", 1)]));
    // LDS: 16-bit at lane*2, 8-bit at 256 + lane; after the barrier read the neighbour's (lane ^ 1)
    code.extend(ds("DS_WRITE_B16", &[("ADDR", 2), ("DATA0", 34)]));
    code.extend(ds("DS_WRITE_B8", &[("ADDR", 4), ("DATA0", 36)]));
    code.extend(encode("S_BARRIER", "ENC_SOPP", &[], None).unwrap());
    code.extend(ds("DS_READ_U16", &[("VDST", 49), ("ADDR", 5)]));
    code.extend(ds("DS_READ_I16", &[("VDST", 50), ("ADDR", 5)]));
    code.extend(ds("DS_READ_U8", &[("VDST", 51), ("ADDR", 6)]));
    code.extend(ds("DS_READ_I8", &[("VDST", 52), ("ADDR", 6)]));
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[24..28].copy_from_slice(&[0x700, 0, 0x200, 0]);
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[2] = 2 * l32;
        r[3] = 0x80 + l32;
        r[4] = 256 + l32;
        r[5] = 2 * (l32 ^ 1);
        r[6] = 256 + (l32 ^ 1);
        r[30] = 0;
        r[31] = l32 + 1;
        r[32] = f(l as f32 - 20.5);
        r[33] = (l as i32 - 30) as u32;
        r[34] = 0xAB00 + l32;
        r[35] = 1;
        r[36] = 0x90 + l32;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x604 / 4] = 1000;
    mem[0x608 / 4] = f32::NEG_INFINITY.to_bits();
    mem[0x60C / 4] = f32::INFINITY.to_bits();
    mem[0x610 / 4] = (-100i32) as u32;
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.mem[0x600 / 4], 64 * 65 / 2, "global_atomic_add");
    let mut olds: Vec<u32> = r.vgpr.iter().map(|v| v[40]).collect();
    olds.sort_unstable();
    olds.dedup();
    assert_eq!(olds.len(), 64, "global_atomic_add with GLC returns 64 distinct old values");
    assert_eq!(r.mem[0x604 / 4], 1000 - 64, "global_atomic_sub");
    assert_eq!(f32::from_bits(r.mem[0x608 / 4]), 42.5, "global_atomic_fmax");
    assert_eq!(f32::from_bits(r.mem[0x60C / 4]), -20.5, "global_atomic_fmin");
    assert_eq!(r.mem[0x610 / 4] as i32, 33, "global_atomic_smax");
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let (s, b) = (0xAB00 + l as u32, 0x90 + l as u32);
        let n = l ^ 1;
        assert_eq!((r.mem[(0x700 + 2 * l) / 4] >> (16 * (l % 2))) & 0xFFFF, s, "lane {l}: buffer_store_short");
        assert_eq!((r.mem[(0x780 + l) / 4] >> (8 * (l % 4))) & 0xFF, b, "lane {l}: buffer_store_byte");
        assert_eq!((v[45], v[46]), (s, s | 0xFFFF_0000), "lane {l}: buffer_load_ushort / sshort");
        assert_eq!((v[47], v[48]), (b, b | 0xFFFF_FF00), "lane {l}: buffer_load_ubyte / sbyte");
        let (sn, bn) = (0xAB00 + n as u32, 0x90 + n as u32);
        assert_eq!((v[49], v[50]), (sn, sn | 0xFFFF_0000), "lane {l}: ds_read_u16 / i16 of the neighbour");
        assert_eq!((v[51], v[52]), (bn, bn | 0xFFFF_FF00), "lane {l}: ds_read_u8 / i8 of the neighbour");
    }
}

/// The instructions added for code outside the retail pack: global inc / dec / csub /
/// 64-bit add, LDS signed min / max, cmpst, mskor, float add (CAS loops), global sub-dword
/// and d16 accesses, scratch, SAD / lerp, the division helpers, saturating adds, packs, swap.
#[test]
fn gpu_more_atomics_sub_dword_scratch_valu() {
    if !gpu_ready_wave32() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_FLAT_GLBL", f);
    let ds = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_DS", f);
    let vop3 = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_VOP3", f);
    let mut code = Vec::new();
    // global atomics at 0x600 + offset (s[2:3] + v30 = 0)
    code.extend(glbl("GLOBAL_ATOMIC_INC", &[("ADDR", 30), ("DATA", 21), ("SADDR", 2), ("OFFSET", 0)]));
    code.extend(glbl("GLOBAL_ATOMIC_DEC", &[("ADDR", 30), ("DATA", 21), ("SADDR", 2), ("OFFSET", 4)]));
    code.extend(glbl("GLOBAL_ATOMIC_CSUB", &[("ADDR", 30), ("DATA", 35), ("SADDR", 2), ("OFFSET", 8)]));
    code.extend(glbl("GLOBAL_ATOMIC_ADD_X2", &[("ADDR", 30), ("DATA", 38), ("SADDR", 2), ("OFFSET", 16)]));
    code.extend(glbl("GLOBAL_ATOMIC_INC_X2", &[("ADDR", 30), ("DATA", 90), ("SADDR", 2), ("OFFSET", 24)]));
    code.extend(glbl("GLOBAL_ATOMIC_DEC_X2", &[("ADDR", 30), ("DATA", 90), ("SADDR", 2), ("OFFSET", 32)]));
    // LDS: initial values, then the atomics, then read back
    for (a, v) in [(7, 8), (17, 9), (18, 10), (19, 12), (20, 15)] {
        code.extend(ds("DS_WRITE_B32", &[("ADDR", a), ("DATA0", v)]));
    }
    code.extend(enc("S_BARRIER", "ENC_SOPP", &[]));
    code.extend(ds("DS_MAX_I32", &[("ADDR", 7), ("DATA0", 33)]));
    code.extend(ds("DS_MIN_I32", &[("ADDR", 17), ("DATA0", 33)]));
    code.extend(ds("DS_CMPST_RTN_B32", &[("VDST", 53), ("ADDR", 18), ("DATA0", 10), ("DATA1", 11)]));
    code.extend(ds("DS_MSKOR_B32", &[("ADDR", 19), ("DATA0", 13), ("DATA1", 14)]));
    code.extend(ds("DS_ADD_F32", &[("ADDR", 20), ("DATA0", 16)]));
    code.extend(enc("S_BARRIER", "ENC_SOPP", &[]));
    for (k, a) in [7u64, 17, 18, 19, 20].iter().enumerate() {
        code.extend(ds("DS_READ_B32", &[("VDST", 54 + k as u64), ("ADDR", *a)]));
    }
    // global bytes at 0x780 + lane, shorts at 0x700 + 2 · lane (s[2:3] + v + 0x100 / 0x180)
    code.extend(glbl("GLOBAL_STORE_BYTE", &[("ADDR", 3), ("DATA", 36), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_UBYTE", &[("VDST", 59), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_SBYTE", &[("VDST", 60), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_UBYTE_D16_HI", &[("VDST", 61), ("ADDR", 3), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_STORE_SHORT_D16_HI", &[("ADDR", 2), ("DATA", 62), ("SADDR", 2), ("OFFSET", 0x100)]));
    code.extend(glbl("GLOBAL_LOAD_USHORT", &[("VDST", 63), ("ADDR", 2), ("SADDR", 2), ("OFFSET", 0x100)]));
    // scratch (per lane): a dword at 2 · lane + 16, a byte at 2 · lane + 40
    code.extend(enc("SCRATCH_STORE_DWORD", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 11), ("SADDR", 0x7F), ("OFFSET", 16)]));
    code.extend(enc("SCRATCH_STORE_BYTE", "ENC_FLAT_SCRATCH", &[("ADDR", 2), ("DATA", 36), ("SADDR", 0x7F), ("OFFSET", 40)]));
    code.extend(enc("SCRATCH_LOAD_DWORD", "ENC_FLAT_SCRATCH", &[("VDST", 64), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 16)]));
    code.extend(enc("SCRATCH_LOAD_UBYTE", "ENC_FLAT_SCRATCH", &[("VDST", 65), ("ADDR", 2), ("SADDR", 0x7F), ("OFFSET", 40)]));
    // VALU
    let vv = |n: u64| V + n;
    code.extend(vop3("V_SAD_U8", &[("VDST", 66), ("SRC0", vv(22)), ("SRC1", vv(23)), ("SRC2", vv(24))]));
    code.extend(vop3("V_MSAD_U8", &[("VDST", 67), ("SRC0", vv(22)), ("SRC1", vv(25)), ("SRC2", vv(24))]));
    code.extend(vop3("V_LERP_U8", &[("VDST", 68), ("SRC0", vv(22)), ("SRC1", vv(23)), ("SRC2", vv(26))]));
    code.extend(vop3("V_DIV_FIXUP_F32", &[("VDST", 69), ("SRC0", vv(37)), ("SRC1", vv(28)), ("SRC2", vv(27))]));
    code.extend(vop3("V_DIV_FIXUP_F32", &[("VDST", 70), ("SRC0", vv(37)), ("SRC1", 128), ("SRC2", vv(27))]));
    code.extend(vop3("V_DIV_FMAS_F32", &[("VDST", 71), ("SRC0", vv(28)), ("SRC1", vv(27)), ("SRC2", vv(28))]));
    code.extend(enc("V_DIV_SCALE_F32", "VOP3_SDST_ENC", &[("VDST", 72), ("SDST", 106), ("SRC0", vv(27)), ("SRC1", vv(28)), ("SRC2", vv(27))]));
    code.extend(vop3("V_ADD_NC_I32", &[("VDST", 73), ("SRC0", vv(29)), ("SRC1", vv(35)), ("CLAMP", 1)]));
    code.extend(vop3("V_CVT_PK_U16_U32", &[("VDST", 74), ("SRC0", vv(76)), ("SRC1", vv(24))]));
    code.extend(vop3("V_MAD_U32_U16", &[("VDST", 75), ("SRC0", vv(77)), ("SRC1", vv(78)), ("SRC2", vv(35))]));
    code.extend(vop3("V_CVT_PKNORM_U16_F32", &[("VDST", 79), ("SRC0", 242), ("SRC1", 240)]));
    // normalized f16 conversions: v82 = halves (0.5, +1.0), v83 = (−1.0, −0.5)
    code.extend(vop3("V_CVT_PKNORM_U16_F16", &[("VDST", 84), ("SRC0", vv(82)), ("SRC1", vv(83))]));
    code.extend(vop3("V_CVT_PKNORM_I16_F16", &[("VDST", 85), ("SRC0", vv(82)), ("SRC1", vv(83)), ("OP_SEL", 3)]));
    code.extend(vop3("V_CVT_NORM_U16_F16", &[("VDST", 86), ("SRC0", vv(82))]));
    code.extend(vop3("V_CVT_NORM_I16_F16", &[("VDST", 86), ("SRC0", vv(83)), ("OP_SEL", 9)]));
    code.extend(enc("V_CVT_NORM_U16_F16", "ENC_VOP1", &[("VDST", 87), ("SRC0", 242)]));
    code.extend(enc("V_SWAP_B32", "ENC_VOP1", &[("VDST", 80), ("SRC0", vv(81))]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l32 = l as u32;
        r[2] = 2 * l32;
        r[3] = 0x80 + l32;
        for (k, a) in [(7, 512u32), (17, 516), (18, 520), (19, 524), (20, 528)] {
            r[k] = a;
        }
        r[8] = (-100i32) as u32;
        r[9] = 100;
        r[10] = 7;
        r[11] = l32 + 1;
        r[12] = 0xFF00;
        r[13] = 0xF0F0;
        r[14] = 0x0101;
        r[15] = 0;
        r[16] = f(0.5);
        r[21] = 10;
        r[22] = 0x1020_3040;
        r[23] = 0x4030_2010;
        r[24] = 5;
        r[25] = 0x0030_2010;
        r[26] = 0x0101_0101;
        r[27] = f(7.0);
        r[28] = f(2.0);
        r[29] = 0x7FFF_FFFF;
        r[30] = 0;
        r[33] = (l as i32 - 30) as u32;
        r[35] = 1;
        r[36] = 0x90 + l32;
        r[37] = f(3.5);
        r[38] = 1;
        r[39] = 0;
        r[61] = 0x1234;
        r[62] = 0xBEEF_1111;
        r[76] = 70000;
        r[77] = 0x1_0003;
        r[78] = 0x2_0004;
        r[80] = 1;
        r[81] = 2;
        r[82] = 0x3C00_3800; // f16 halves: lo +0.5, hi +1.0
        r[83] = 0xB800_BC00; // lo −1.0, hi −0.5
        r[90] = 10;
        r[91] = 0;
    }
    let mut mem = vec![0u32; 1024];
    mem[0x604 / 4] = 5;
    mem[0x608 / 4] = 50;
    mem[0x610 / 4] = 0xFFFF_FFF0;
    mem[0x620 / 4] = 5;
    let r = runner::run_code(&code, &st, &mem).unwrap();
    assert_eq!(r.mem[0x600 / 4], 64 % 11, "global_atomic_inc wraps at 10");
    assert_eq!(r.mem[0x604 / 4], 7, "global_atomic_dec wraps to 10 below 0");
    assert_eq!(r.mem[0x608 / 4], 0, "global_atomic_csub clamps at 0");
    assert_eq!((r.mem[0x610 / 4], r.mem[0x614 / 4]), (0x30, 1), "global_atomic_add_x2 carries into the high dword");
    assert_eq!((r.mem[0x618 / 4], r.mem[0x61C / 4]), (64 % 11, 0), "global_atomic_inc_x2 wraps at 10");
    assert_eq!((r.mem[0x620 / 4], r.mem[0x624 / 4]), (7, 0), "global_atomic_dec_x2 wraps to 10 below 0");
    assert_eq!(r.vgpr.iter().filter(|v| v[53] == 7).count(), 1, "ds_cmpst: one lane finds the compare value");
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let b = 0x90 + l as u32;
        assert_eq!((v[54] as i32, v[55] as i32), (33, -30), "lane {l}: ds_max_i32 / ds_min_i32");
        assert!((1..=64).contains(&v[56]), "lane {l}: ds_cmpst stored a lane's value");
        assert_eq!((v[57], f32::from_bits(v[58])), (0x0F01, 32.0), "lane {l}: ds_mskor_b32 / ds_add_f32");
        assert_eq!((v[59], v[60], v[61]), (b, b | 0xFFFF_FF00, 0x1234 | b << 16), "lane {l}: global ubyte / sbyte / ubyte_d16_hi");
        assert_eq!(v[63], 0xBEEF, "lane {l}: global_store_short_d16_hi + load_ushort");
        assert_eq!((v[64], v[65]), (l as u32 + 1, b), "lane {l}: scratch dword / byte");
        assert_eq!((v[66], v[67], v[68]), (133, 85, 0x2828_2828), "lane {l}: v_sad_u8 / v_msad_u8 / v_lerp_u8");
        assert_eq!((f32::from_bits(v[69]), f32::from_bits(v[70]), f32::from_bits(v[71])), (3.5, f32::INFINITY, 16.0), "lane {l}: div_fixup / fixup(x/0) / div_fmas");
        assert_eq!((f32::from_bits(v[72]), v[73]), (7.0, 0x7FFF_FFFF), "lane {l}: v_div_scale_f32 / v_add_nc_i32 clamp");
        assert_eq!((v[74], v[75], v[79]), (0x0005_FFFF, 13, 0x8000_FFFF), "lane {l}: cvt_pk_u16_u32 / mad_u32_u16 / cvt_pknorm_u16_f32");
        assert_eq!((v[84], v[85], v[86], v[87]), (0x0000_8000, 0xC000_7FFF, 0xC000_8000, 0xFFFF), "lane {l}: cvt norm f16 (unorm/snorm, packed, hi halves)");
        assert_eq!((v[80], v[81]), (2, 1), "lane {l}: v_swap_b32");
    }
    assert_eq!(r.sgpr[106] | r.sgpr[107], 0, "v_div_scale_f32: VCC 0 (no post-scale)");
}

/// *_atomic_fcmpswap_x2 on hardware: a bitwise 64-bit compare-swap. A compare value equal
/// as a float but different in bits (-0.0 vs +0.0) must NOT swap; GLC returns the old qword.
/// Per-lane cells are deterministic; the three shared cells end in the unique winner's value.
#[test]
fn gpu_atomic_fcmpswap_x2() {
    if !gpu_ready_wave32() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let glbl = |n: &'static str, f: &[(&str, u64)]| enc(n, "ENC_FLAT_GLBL", f);
    let mut code = Vec::new();
    // buffer: V# s[8:11] = raw base 0x600; per-lane cell at base + voff (OFFEN), voff = 8·lane
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 4), ("VADDR", 2), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1), ("GLC", 1)]));
    // flat: cell at v14 (0x800 + 8·lane)
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 12), ("ADDR", 14), ("DATA", 16), ("GLC", 1)]));
    // global: cell at s2 + v22 (0xA00 + 8·lane)
    code.extend(glbl("GLOBAL_ATOMIC_FCMPSWAP_X2", &[("VDST", 20), ("ADDR", 22), ("DATA", 24), ("SADDR", 2), ("GLC", 1)]));
    // one cell shared by all 64 lanes per family: buffer 0xC00 (V# s[12:15]), flat 0xE00,
    // global s3 + v9 (0x1200)
    code.extend(enc("BUFFER_ATOMIC_FCMPSWAP_X2", "ENC_MUBUF", &[("VDATA", 30), ("VADDR", 8), ("SRSRC", 3), ("SOFFSET", 128), ("GLC", 1)]));
    code.extend(enc("FLAT_ATOMIC_FCMPSWAP_X2", "ENC_FLAT", &[("VDST", 40), ("ADDR", 10), ("DATA", 34), ("GLC", 1)]));
    code.extend(glbl("GLOBAL_ATOMIC_FCMPSWAP_X2", &[("VDST", 44), ("ADDR", 9), ("DATA", 46), ("SADDR", 3), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0xA00;
    st.sgpr[3] = 0x1200;
    st.sgpr[8..12].copy_from_slice(&[0x600, 0, 0x200, 0]); // V#: base, stride/flags, records, –
    st.sgpr[12..16].copy_from_slice(&[0xC00, 0, 0x200, 0]);
    const INIT: u64 = 0x3FF8_0000_0000_0000; // f64 bits of 1.5
    const NEG_ZERO: u64 = 0x8000_0000_0000_0000; // -0.0: == +0.0 numerically, different bits
    let newv = |tag: u64, l: u64| tag << 48 | l;
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        let l64 = l as u64;
        r[2] = 8 * l as u32; // buffer offset -> cell 0x600 + 8·lane
        r[14] = 0x800 + 8 * l as u32;
        r[22] = 8 * l as u32; // + s2 -> 0xA00 + 8·lane
        r[10] = 0xE00; // flat shared cell (all lanes)
        // even lanes: cell = INIT, cmp = INIT -> swap happens; odd lanes: cell = +0.0, cmp =
        // -0.0 -> bitwise different -> no swap (a numeric compare would swap)
        let cmp = if l % 2 == 0 { INIT } else { NEG_ZERO };
        let put = |r: &mut Vec<u32>, at: usize, v: u64| {
            r[at] = v as u32;
            r[at + 1] = (v >> 32) as u32;
        };
        put(r, 4, newv(0xB000, l64)); // buffer: new at [4:5], cmp at [6:7]
        put(r, 6, cmp);
        put(r, 16, newv(0xF000, l64));
        put(r, 18, cmp);
        put(r, 24, newv(0xA000, l64));
        put(r, 26, cmp);
        put(r, 30, newv(0x5B00, l64)); // shared cells: every lane compares against INIT
        put(r, 32, INIT);
        put(r, 34, newv(0x5F00, l64));
        put(r, 36, INIT);
        put(r, 46, newv(0x5A00, l64));
        put(r, 48, INIT);
    }
    let mut mem = vec![0u32; 0x1400 / 4];
    for l in 0..64usize {
        let b = if l % 2 == 0 { INIT } else { 0 };
        for a in [0x600 + 8 * l, 0x800 + 8 * l, 0xA00 + 8 * l] {
            mem[a / 4] = b as u32;
            mem[a / 4 + 1] = (b >> 32) as u32;
        }
    }
    for a in [0xC00, 0xE00, 0x1200] {
        mem[a / 4] = INIT as u32;
        mem[a / 4 + 1] = (INIT >> 32) as u32;
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let old = |v: &[u32], at: usize| v[at] as u64 | (v[at + 1] as u64) << 32;
    let cell = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
    for l in 0..64usize {
        let v = &r.vgpr[l];
        let even = l % 2 == 0;
        let l64 = l as u64;
        // per-lane cells: swapped iff cmp == cell bits
        let (bs, fs, gs) = if even { (newv(0xB000, l64), newv(0xF000, l64), newv(0xA000, l64)) } else { (0, 0, 0) };
        let cmp_old = if even { INIT } else { 0 }; // the cell's pre-swap value
        assert_eq!(cell(0x600 + 8 * l), bs, "lane {l}: buffer cell");
        assert_eq!(cell(0x800 + 8 * l), fs, "lane {l}: flat cell");
        assert_eq!(cell(0xA00 + 8 * l), gs, "lane {l}: global cell");
        // GLC old value: VDATA in place for buffer, VDST for flat / global
        assert_eq!(old(v, 4), cmp_old, "lane {l}: buffer GLC old");
        assert_eq!(old(v, 12), cmp_old, "lane {l}: flat GLC old");
        assert_eq!(old(v, 20), cmp_old, "lane {l}: global GLC old");
        // the compare register half is untouched
        assert_eq!(old(v, 6), if even { INIT } else { NEG_ZERO }, "lane {l}: buffer cmp regs");
    }
    // shared cells: exactly one lane sees INIT (the winner); the cell holds its value
    for (i, (dst, a, tag)) in [(30, 0xC00usize, 0x5B00u64), (40, 0xE00, 0x5F00), (44, 0x1200, 0x5A00)].iter().enumerate() {
        let winners: Vec<usize> = r.vgpr.iter().enumerate().filter(|(_, v)| old(v, *dst) == INIT).map(|(l, _)| l).collect();
        assert_eq!(winners.len(), 1, "shared cell {i}: exactly one lane wins the swap");
        assert_eq!(cell(*a), newv(*tag, winners[0] as u64), "shared cell {i}: the winner's value landed");
    }
}

/// 32-bit float atomic min / max in memory with a -0.0 operand. The translation turns them
/// into integer atomics chosen by the operand's sign; -0.0 has its sign bit set, so it has to
/// take the negative path: fmin(-5, -0) = -5, fmax(-5, -0) = -0, fmin(3, -0) = -0,
/// fmax(3, -0) = 3. Both code paths: global (flat / global) and buffer.
#[test]
fn gpu_atomic_fmin_fmax_negative_zero() {
    if !gpu_ready_wave32() {
        return;
    }
    let glbl = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_FLAT_GLBL", f, None).unwrap();
    let mubuf = |n: &'static str, f: &[(&str, u64)]| encode(n, "ENC_MUBUF", f, None).unwrap();
    let mut code = Vec::new();
    // global: s2 + v30 (= 0) + offset; buffer: V# s[8:11], raw base 0x700, + offset
    for (op, off) in [("FMIN", 0u64), ("FMAX", 4), ("FMIN", 8), ("FMAX", 12)] {
        let (g, b) = match op {
            "FMIN" => ("GLOBAL_ATOMIC_FMIN", "BUFFER_ATOMIC_FMIN"),
            _ => ("GLOBAL_ATOMIC_FMAX", "BUFFER_ATOMIC_FMAX"),
        };
        code.extend(glbl(g, &[("ADDR", 30), ("DATA", 32), ("SADDR", 2), ("OFFSET", off)]));
        code.extend(mubuf(b, &[("VDATA", 32), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", off)]));
    }
    code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    let mut st = KernelState::default();
    st.sgpr[2] = 0x600;
    st.sgpr[8..12].copy_from_slice(&[0x700, 0, 0x200, 0]);
    for r in st.vgpr.iter_mut() {
        r[30] = 0;
        r[32] = (-0.0f32).to_bits();
    }
    let mut mem = vec![0u32; 0x800 / 4];
    for base in [0x600usize, 0x700] {
        mem[base / 4] = (-5.0f32).to_bits();
        mem[base / 4 + 1] = (-5.0f32).to_bits();
        mem[base / 4 + 2] = 3.0f32.to_bits();
        mem[base / 4 + 3] = 3.0f32.to_bits();
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    for (family, base) in [("global", 0x600usize), ("buffer", 0x700)] {
        let cell = |k: usize| r.mem[base / 4 + k];
        assert_eq!(cell(0), (-5.0f32).to_bits(), "{family}: fmin(-5, -0) = -5");
        assert_eq!(cell(1), (-0.0f32).to_bits(), "{family}: fmax(-5, -0) = -0");
        assert_eq!(cell(2), (-0.0f32).to_bits(), "{family}: fmin(3, -0) = -0");
        assert_eq!(cell(3), 3.0f32.to_bits(), "{family}: fmax(3, -0) = 3");
    }
}

/// *_atomic_fmin_x2 / fmax_x2 on hardware, all three families. Six cells each: 64 lanes
/// race on a max (from -inf) and a min (from +inf) of lane - 20.5, then the -0.0 cases of
/// the f32 test. A seventh, per-lane cell checks the GLC return (global).
#[test]
fn gpu_atomic_fmin_fmax_x2() {
    if !gpu_ready_wave32() {
        return;
    }
    let enc = |n: &'static str, e: &'static str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"));
    let mut code = Vec::new();
    // (op, data VGPR, cell offset): the lane values at v[32:33], -0.0 at v[34:35]
    let ops = [("FMAX", 32u64, 0u64), ("FMIN", 32, 8), ("FMIN", 34, 16), ("FMAX", 34, 24), ("FMIN", 34, 32), ("FMAX", 34, 40)];
    for (op, data, off) in ops {
        let x2 = |family: &str| -> &'static str {
            match (family, op) {
                ("BUFFER", "FMIN") => "BUFFER_ATOMIC_FMIN_X2",
                ("BUFFER", _) => "BUFFER_ATOMIC_FMAX_X2",
                ("FLAT", "FMIN") => "FLAT_ATOMIC_FMIN_X2",
                ("FLAT", _) => "FLAT_ATOMIC_FMAX_X2",
                (_, "FMIN") => "GLOBAL_ATOMIC_FMIN_X2",
                _ => "GLOBAL_ATOMIC_FMAX_X2",
            }
        };
        // buffer: V# s[8:11], raw base 0x800; flat: address v[36:37] = 0x900; global: s2 = 0xA00
        code.extend(enc(x2("BUFFER"), "ENC_MUBUF", &[("VDATA", data), ("VADDR", 30), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", off)]));
        code.extend(enc(x2("FLAT"), "ENC_FLAT", &[("ADDR", 36), ("DATA", data), ("OFFSET", off)]));
        code.extend(enc(x2("GLOBAL"), "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", data), ("SADDR", 2), ("OFFSET", off)]));
    }
    // per-lane cell at s3 + v38 (0xB00 + 8·lane): max(1.0, 2.0), the old 1.0 back in v[40:41]
    code.extend(enc("GLOBAL_ATOMIC_FMAX_X2", "ENC_FLAT_GLBL", &[("VDST", 40), ("ADDR", 38), ("DATA", 42), ("SADDR", 3), ("GLC", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let mut st = KernelState::default();
    st.sgpr[2] = 0xA00;
    st.sgpr[3] = 0xB00;
    st.sgpr[8..12].copy_from_slice(&[0x800, 0, 0x100, 0]);
    let put = |r: &mut Vec<u32>, at: usize, v: u64| {
        r[at] = v as u32;
        r[at + 1] = (v >> 32) as u32;
    };
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[30] = 0;
        put(r, 32, (l as f64 - 20.5).to_bits());
        put(r, 34, (-0.0f64).to_bits());
        put(r, 36, 0x900);
        r[38] = 8 * l as u32;
        put(r, 42, 2.0f64.to_bits());
    }
    let init = [f64::NEG_INFINITY, f64::INFINITY, -5.0, -5.0, 3.0, 3.0];
    let mut mem = vec![0u32; 0xE00 / 4];
    let set = |mem: &mut Vec<u32>, a: usize, v: f64| {
        mem[a / 4] = v.to_bits() as u32;
        mem[a / 4 + 1] = (v.to_bits() >> 32) as u32;
    };
    for base in [0x800usize, 0x900, 0xA00] {
        for (k, v) in init.iter().enumerate() {
            set(&mut mem, base + 8 * k, *v);
        }
    }
    for l in 0..64usize {
        set(&mut mem, 0xB00 + 8 * l, 1.0);
    }
    let r = runner::run_code(&code, &st, &mem).unwrap();
    let cell = |a: usize| r.mem[a / 4] as u64 | (r.mem[a / 4 + 1] as u64) << 32;
    let want = [42.5f64.to_bits(), (-20.5f64).to_bits(), (-5.0f64).to_bits(), (-0.0f64).to_bits(), (-0.0f64).to_bits(), 3.0f64.to_bits()];
    let what = ["max of the lanes", "min of the lanes", "fmin(-5, -0) = -5", "fmax(-5, -0) = -0", "fmin(3, -0) = -0", "fmax(3, -0) = 3"];
    for (family, base) in [("buffer", 0x800usize), ("flat", 0x900), ("global", 0xA00)] {
        for k in 0..6 {
            assert_eq!(f64::from_bits(cell(base + 8 * k)), f64::from_bits(want[k]), "{family}: {}", what[k]);
            assert_eq!(cell(base + 8 * k), want[k], "{family}: {} (bits)", what[k]);
        }
    }
    for (l, v) in r.vgpr.iter().enumerate() {
        assert_eq!(cell(0xB00 + 8 * l), 2.0f64.to_bits(), "lane {l}: per-lane fmax_x2");
        let old = v[40] as u64 | (v[41] as u64) << 32;
        assert_eq!(old, 1.0f64.to_bits(), "lane {l}: GLC returns the old f64");
    }
}

/// `XCL_BLOB=<console CS container>`: HLSL size and shape of its translation.
#[test]
#[ignore]
fn blob_hlsl_shape() {
    let Some(p) = std::env::var_os("XCL_BLOB") else { return };
    let blob = std::fs::read(p).unwrap();
    let (h, _) = super::pso::console_to_hlsl(&blob, super::pso::Stage::Compute).unwrap();
    println!("HLSL {} KB, pc-dispatch: {}, if: {}, for(;;): {}", h.len() / 1024, h.matches("pc ==").count(), h.matches("if (!(").count(), h.matches("for (;;)").count());
    if let Some(out) = std::env::var_os("XCL_HLSL_OUT") {
        std::fs::write(out, &h).unwrap();
    }
}

/// `XCL_BLOB=<console CS>` `XCL_DXC_ARGS="-O3"`: DXC time and DXIL size for its HLSL (this
/// process is capped at 8 GB of commit).
#[test]
#[ignore]
fn blob_dxc_cost() {
    let Some(p) = std::env::var_os("XCL_BLOB") else { return };
    unsafe {
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = 8 << 30;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
    let blob = std::fs::read(p).unwrap();
    let (h, profile) = super::pso::console_to_hlsl(&blob, super::pso::Stage::Compute).unwrap();
    let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
    let extra: Vec<&str> = args.split_whitespace().collect();
    let t0 = std::time::Instant::now();
    let r = super::dxc::compile(&h, "main", profile, &extra);
    println!("HLSL {} KB, DXC {args}: {:.1} s → {}", h.len() / 1024, t0.elapsed().as_secs_f32(), match r {
        Ok(b) => format!("DXIL {} KB", b.len() / 1024),
        Err(e) => format!("error: {}", e.chars().take(200).collect::<String>()),
    });
}

/// `XCL_RAW=<file of raw RDNA2 code>`: its disassembly (reverse-engineering helper).
#[test]
#[ignore]
fn raw_rdna2_listing() {
    let Some(p) = std::env::var_os("XCL_RAW") else { return };
    let code = std::fs::read(p).unwrap();
    for l in crate::gpu::rdna2::disassemble(&code, 0, false) {
        println!("  {:06x}: {}", l.pc, l.text);
    }
}

/// Every FSSM variant of the pack through the translator (no GPU): how many translate
/// completely, per stage, and which instructions are missing (weighted by shaders).
/// `XCL_SAMPLE_RPF=…\shaders.rpf`, `XCL_SURVEY_EVERY=N` (every N-th record, default 1),
/// `XCL_SURVEY_FIND=text` (only records whose name contains it), `XCL_SURVEY_SKIP=a|b` (leave out
/// records whose name contains one of these), `XCL_SURVEY_TRACE=1` (start / time / result per program);
/// `XCL_SURVEY_DXC=1` also compiles each one with DXC (4 threads, process capped at 8 GB):
/// failures and compile times.
#[test]
#[ignore]
fn retail_translation_survey() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let every: usize = std::env::var("XCL_SURVEY_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let data = std::fs::read(path).unwrap();
    // XCL_SURVEY_SEED=<n> or "random": a random 1/every of the records (splitmix64), else every N-th
    let seed = std::env::var("XCL_SURVEY_SEED").ok().map(|v| {
        v.parse::<u64>().unwrap_or_else(|_| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64))
    });
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut all = crate::gpu::pack::scan(&data, 0x40);
    if let Ok(find) = std::env::var("XCL_SURVEY_FIND") {
        all.retain(|r| r.name.contains(&find));
        println!("{} records whose name contains {find:?}", all.len());
    }
    if let Ok(skip) = std::env::var("XCL_SURVEY_SKIP") {
        all.retain(|r| !skip.split('|').any(|s| !s.is_empty() && r.name.contains(s)));
        println!("{} records after skipping {skip:?}", all.len());
    }
    let records: Vec<_> = match seed {
        Some(sd) => {
            println!("random sample: seed {sd}");
            all.into_iter().enumerate().filter(|(i, _)| mix(sd ^ (*i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) % every as u64 == 0).map(|(_, r)| r).collect()
        }
        None => all.into_iter().step_by(every).collect(),
    };
    #[derive(Default)]
    struct Acc {
        // stage → (variants, complete)
        stages: BTreeMap<String, (usize, usize)>,
        // "mnemonic: reason" → shaders
        missing: BTreeMap<String, usize>,
        // error / no interface → shaders (with one example name)
        errors: BTreeMap<String, (usize, String)>,
        instructions: usize,
        // DXC: seconds per variant, failures "stage: first line" → (count, example)
        dxc_times: Vec<(f64, String)>,
        // DXC seconds / variants: fully structured, with `pc` guards
        dxc_structured: (f64, usize),
        dxc_hybrid: (f64, usize),
        hybrid: usize,
        hybrid_names: Vec<(usize, String)>,
        dxc_errors: BTreeMap<String, (usize, String)>,
    }
    let with_dxc = std::env::var_os("XCL_SURVEY_DXC").is_some() && super::dxc::location().is_ok();
    if with_dxc {
        unsafe {
            use windows_sys::Win32::System::JobObjects::*;
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            info.ProcessMemoryLimit = 8 << 30;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
            AssignProcessToJobObject(job, GetCurrentProcess());
        }
    }
    let workers = if with_dxc { 4 } else { std::thread::available_parallelism().map_or(4, |n| n.get()) };
    let acc = Mutex::new(Acc::default());
    let next = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                for f in &sh.fssm {
                    let hw = f.stages.first().map_or(0, |s| s.hw);
                    let stage = format!("{} ({})", sh.stage, crate::gpu::xbox::hw_stage_name(hw));
                    let res = std::panic::catch_unwind(|| -> Result<(super::Translation, &'static str), String> {
                        let mode = super::interface(&blob, &sh, f).ok_or("no interface")?;
                        let profile = match &mode {
                            Mode::Vertex(_) => "vs_6_6",
                            Mode::Pixel(_) => "ps_6_6",
                            // MS (NGG) and GS run as compute shaders feeding a pass-through mesh shader
                            Mode::Compute(_) | Mode::Mesh(_) | Mode::Geometry(_) | Mode::Kernel => "cs_6_6",
                        };
                        let p = f.programs.first().ok_or("FSSM without a program")?;
                        let code = &f.data[p.offset..p.offset + p.len];
                        // XCL_SURVEY_WINDOWS=N: translate for N memory windows (above 4 GiB)
                        let mem_windows = std::env::var("XCL_SURVEY_WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                        let trace = std::env::var_os("XCL_SURVEY_TRACE").is_some();
                        if trace {
                            eprintln!("translate start {} ({} B)", r.name, code.len());
                        }
                        let t1 = std::time::Instant::now();
                        let t = super::translate(code, &Options { mode, wave64: super::detect_wave64(code), mem_windows, ..Default::default() });
                        if trace {
                            let state = match &t {
                                Ok(t) if t.complete() => "complete".to_string(),
                                Ok(t) => format!("partial {:?}", t.unsupported_why.keys().collect::<Vec<_>>()),
                                Err(e) => format!("error {e}"),
                            };
                            eprintln!("translate done {} {:.2} s {state}", r.name, t1.elapsed().as_secs_f64());
                        }
                        Ok((t?, profile))
                    });
                    // DXC outside the lock
                    let guarded = matches!(&res, Ok(Ok((t, _))) if t.hlsl.contains("pc =="));
                    // XCL_SURVEY_ONLY=a|b: DXC only for programs whose HLSL contains one of these
                    let only = std::env::var("XCL_SURVEY_ONLY").ok();
                    let wanted = |h: &str| only.as_ref().is_none_or(|o| o.split('|').any(|x| h.contains(x)));
                    let dxc = match (&res, with_dxc) {
                        (Ok(Ok((t, profile))), true) if wanted(&t.hlsl) => {
                            if std::env::var_os("XCL_SURVEY_TRACE").is_some() {
                                eprintln!("dxc start {} ({} KB)", r.name, t.hlsl.len() / 1024);
                            }
                            let t1 = std::time::Instant::now();
                            let r = super::dxc::compile(&t.hlsl, "main", profile, &[]);
                            // the specific message (validation failures list it after "validation errors")
                            let first = |e: &str| {
                                e.lines().find(|l| l.contains("error") && !l.contains("validation errors")).or_else(|| e.lines().find(|l| l.contains("error"))).unwrap_or(e.lines().next().unwrap_or("")).to_string()
                            };
                            Some((t1.elapsed().as_secs_f64(), r.err().map(|e| first(&e))))
                        }
                        _ => None,
                    };
                    let mut a = acc.lock().unwrap();
                    let e = a.stages.entry(stage.clone()).or_default();
                    e.0 += 1;
                    let error = |a: &mut Acc, msg: String| {
                        let x = a.errors.entry(format!("{stage}: {msg}")).or_insert((0, r.name.clone()));
                        x.0 += 1;
                    };
                    a.hybrid += guarded as usize;
                    if let (true, Ok(Ok((t, _)))) = (guarded, &res) {
                        a.hybrid_names.push((t.instructions, r.name.clone()));
                    }
                    if let Some((secs, err)) = dxc {
                        a.dxc_times.push((secs, r.name.clone()));
                        let b = if guarded { &mut a.dxc_hybrid } else { &mut a.dxc_structured };
                        b.0 += secs;
                        b.1 += 1;
                        if let Some(e) = err {
                            // drop the file:line:col prefix so equal errors group
                            let e = e.split("error:").nth(1).unwrap_or(&e).trim().to_string();
                            let x = a.dxc_errors.entry(format!("{stage}: {e}")).or_insert((0, r.name.clone()));
                            x.0 += 1;
                        }
                    }
                    match res {
                        Ok(Ok((t, _))) => {
                            a.instructions += t.instructions;
                            if t.complete() {
                                a.stages.get_mut(&stage).unwrap().1 += 1;
                            }
                            for k in t.unsupported_why.keys() {
                                *a.missing.entry(k.clone()).or_default() += 1;
                            }
                        }
                        Ok(Err(m)) => error(&mut a, m),
                        Err(_) => error(&mut a, "translator panic".into()),
                    }
                }
            });
        }
    });
    let _ = std::panic::take_hook();
    let a = acc.into_inner().unwrap();
    let (all, done): (usize, usize) = a.stages.values().fold((0, 0), |s, v| (s.0 + v.0, s.1 + v.1));
    println!("records: {} (1 in {every}), FSSM variants: {all}, fully translated: {done} ({:.1}%), instructions: {}, time {:.1} s", records.len(), 100.0 * done as f64 / all.max(1) as f64, a.instructions, t0.elapsed().as_secs_f64());
    println!("with pc blocks (not fully structured): {} / {all}", a.hybrid);
    let mut hn = a.hybrid_names;
    hn.sort();
    println!("  smallest: {:?}", hn.iter().take(6).collect::<Vec<_>>());
    for (st, (n, ok)) in &a.stages {
        println!("  {st:<14} {ok:>6} / {n:<6} ({:.1}%)", 100.0 * *ok as f64 / (*n).max(1) as f64);
    }
    let mut missing: Vec<_> = a.missing.into_iter().collect();
    missing.sort_by(|x, y| y.1.cmp(&x.1));
    println!("missing instructions (in how many variants):");
    for (k, n) in missing.iter().take(40) {
        println!("  {n:>6}  {k}");
    }
    let mut errors: Vec<_> = a.errors.into_iter().collect();
    errors.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
    println!("errors:");
    for (k, (n, ex)) in errors.iter().take(20) {
        println!("  {n:>6}  {k}   (e.g. {ex})");
    }
    if with_dxc {
        let mut t = a.dxc_times;
        t.sort_by(|x, y| x.0.total_cmp(&y.0));
        let q = |f: f64| t.get(((t.len() as f64 * f) as usize).min(t.len().saturating_sub(1))).map_or(0.0, |x| x.0);
        let failed: usize = a.dxc_errors.values().map(|v| v.0).sum();
        println!(
            "DXC structured: {} variants, {:.0} s (mean {:.2} s); with pc blocks: {} variants, {:.0} s (mean {:.2} s)",
            a.dxc_structured.1,
            a.dxc_structured.0,
            a.dxc_structured.0 / a.dxc_structured.1.max(1) as f64,
            a.dxc_hybrid.1,
            a.dxc_hybrid.0,
            a.dxc_hybrid.0 / a.dxc_hybrid.1.max(1) as f64
        );
        println!("DXC: {} compiled, {failed} errors; time median {:.2} s, p90 {:.2} s, max {:.1} s ({}), total {:.0} s", t.len(), q(0.5), q(0.9), t.last().map_or(0.0, |x| x.0), t.last().map_or("", |x| x.1.as_str()), t.iter().map(|x| x.0).sum::<f64>());
        for (secs, name) in t.iter().rev().take(10) {
            println!("  {secs:>6.2} s  {name}");
        }
        let mut errs: Vec<_> = a.dxc_errors.into_iter().collect();
        errs.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
        for (k, (n, ex)) in errs.iter().take(20) {
            println!("  {n:>6}  {k}   (e.g. {ex})");
        }
    }
}

/// Shape of the HLSL and DXC cost of named retail shaders (reverse engineering of compile
/// times): `XCL_SAMPLE_RPF`, `XCL_NAMES=a,b` (substring of the record name), `XCL_DXC_ARGS`,
/// `XCL_HLSL_OUT=<folder>` (writes the HLSL). The process is capped at 8 GB.
#[test]
#[ignore]
fn retail_shader_cost() {
    let (Ok(path), Ok(names)) = (std::env::var("XCL_SAMPLE_RPF"), std::env::var("XCL_NAMES")) else { return };
    unsafe {
        use windows_sys::Win32::System::JobObjects::*;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        info.ProcessMemoryLimit = 8 << 30;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const _, std::mem::size_of_val(&info) as u32);
        AssignProcessToJobObject(job, GetCurrentProcess());
    }
    let args = std::env::var("XCL_DXC_ARGS").unwrap_or_default();
    let extra: Vec<&str> = args.split_whitespace().collect();
    let data = std::fs::read(path).unwrap();
    for r in crate::gpu::pack::scan(&data, 0x40) {
        if !names.split(',').any(|n| !n.is_empty() && r.name.contains(n)) {
            continue;
        }
        let Some(blob) = crate::gpu::pack::inflate(&data, &r) else { continue };
        let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
        // effect-qualified names (`effect:CS_…`) are not file names (`:` opens an NTFS stream)
        let stem = r.name.replace(['/', '\\', ':'], "_");
        for f in &sh.fssm {
            let Some(mode) = super::interface(&blob, &sh, f) else { continue };
            let profile = match &mode {
                Mode::Vertex(_) => "vs_6_6",
                Mode::Pixel(_) => "ps_6_6",
                _ => "cs_6_6",
            };
            let p = f.programs[0];
            let code = &f.data[p.offset..p.offset + p.len];
            let t = super::translate(code, &Options { mode, wave64: super::detect_wave64(code), ..Default::default() }).unwrap();
            let owned = t.hlsl.clone();
            let h = &owned;
            if let Ok(dir) = std::env::var("XCL_HLSL_OUT") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.hlsl")), h);
                eprintln!("HLSL written: {} ({} KB)", r.name, h.len() / 1024);
            }
            let t0 = std::time::Instant::now();
            let r2 = super::dxc::compile(h, "main", profile, &extra);
            let secs = t0.elapsed().as_secs_f64();
            if let (Err(_), Ok(dir)) = (&r2, std::env::var("XCL_HLSL_OUT")) {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.failed.hlsl")), h);
            }
            println!(
                "{} [{profile}] instructions {}, HLSL {} KB, lines {}, pc-dispatch {}, if {}, for(;;) {}, select {} → DXC {:.2} s, {}",
                r.name,
                t.instructions,
                h.len() / 1024,
                h.lines().count(),
                h.matches("pc ==").count(),
                h.matches("if (").count(),
                h.matches("for (;;)").count(),
                h.matches("select(").count(),
                secs,
                r2.as_ref().map(|b| format!("DXIL {} KB", b.len() / 1024)).unwrap_or_else(|e| e.lines().next().unwrap_or("").to_string())
            );
            if let Ok(dir) = std::env::var("XCL_HLSL_OUT") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.hlsl")), h);
            }
        }
    }
}

/// ISA coverage: every instruction the decoder knows (RDNA2 spec + console extensions), one
/// at a time with zero operands, through the translator (compute kernel mode, then pixel
/// mode for graphics-only ones). Per encoding: translated / all, and the missing ones.
#[test]
#[ignore]
fn isa_coverage() {
    use crate::gpu::rdna2::{all_instructions, encode};
    use crate::gpu::rdna2_spec::ENCODINGS;
    use std::collections::BTreeMap;
    let mut per_enc: BTreeMap<&str, (usize, usize, Vec<&str>)> = BTreeMap::new();
    let try_mode = |code: &[u8], mode: Mode| match translate(code, &Options { mode, ..Default::default() }) {
        Ok(t) => t.complete(),
        Err(_) => false,
    };
    std::panic::set_hook(Box::new(|_| {}));
    for inst in all_instructions() {
        // the first plain encoding (not an SDWA / DPP / literal variant)
        let Some(ie) = inst.encs.iter().find(|e| ENCODINGS[e.encoding as usize].primary) else { continue };
        let enc = ENCODINGS[ie.encoding as usize].name;
        let lit = inst.always_literal.then_some(0);
        // operands that must not be 0: MIMG channel mask, VOP3 SDST (VCC)
        let fields: &[(&str, u64)] = match enc {
            "ENC_MIMG" => &[("DMASK", 1)],
            "VOP3_SDST_ENC" => &[("SDST", 106)],
            _ if inst.name == "V_SWAPREL_B32" => &[("SRC0", V)],
            _ => &[],
        };
        let Ok(mut code) = encode(inst.name, enc, fields, lit) else { continue };
        code.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        let ok = std::panic::catch_unwind(|| {
            try_mode(&code, Mode::Kernel) || try_mode(&code, Mode::Pixel(super::PsInterface { inputs: vec![], outputs: vec![], input_ena: 0x7F, attrs: None }))
        })
        .unwrap_or(false);
        let e = per_enc.entry(enc.trim_start_matches("ENC_")).or_default();
        e.1 += 1;
        if ok {
            e.0 += 1;
        } else {
            e.2.push(inst.name);
        }
    }
    let _ = std::panic::take_hook();
    let (ok, all) = per_enc.values().fold((0, 0), |s, v| (s.0 + v.0, s.1 + v.1));
    println!("RDNA2 instructions (spec + console): {all}, translated: {ok} ({:.1}%)", 100.0 * ok as f64 / all.max(1) as f64);
    for (enc, (ok, all, missing)) in &per_enc {
        let m: Vec<String> = missing.iter().map(|n| n.to_lowercase()).collect();
        let full = std::env::var_os("XCL_ISA_FULL").is_some();
        println!("  {enc:<12} {ok:>4} / {all:<4} missing: {}", if m.len() > 14 && !full { format!("{} … (+{})", m[..14].join(", "), m.len() - 14) } else { m.join(", ") });
    }
}

/// Census of the cross-lane / control-flow forms the translator special-cases: every
/// `ds_swizzle_b32` offset in the pack, and the instructions around each `s_setpc_b64` /
/// `s_swappc_b64` of the records whose name contains one of `XCL_NAMES` (`a,b`).
#[test]
#[ignore]
fn retail_swizzle_and_setpc_census() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let Ok(path) = std::env::var("XCL_SAMPLE_RPF") else { return };
    let names = std::env::var("XCL_NAMES").unwrap_or_default();
    let data = std::fs::read(path).unwrap();
    let records = crate::gpu::pack::scan(&data, 0x40);
    let swz: Mutex<BTreeMap<u64, (usize, String)>> = Mutex::new(BTreeMap::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(r) = records.get(i) else { break };
                let Some(blob) = crate::gpu::pack::inflate(&data, r) else { continue };
                let Some(sh) = crate::gpu::xbox::parse(&blob) else { continue };
                let show = names.split(',').any(|n| !n.is_empty() && r.name.contains(n));
                for f in &sh.fssm {
                    for p in &f.programs {
                        let code = &f.data[p.offset..p.offset + p.len];
                        if let (true, Ok(dir)) = (show, std::env::var("XCL_RAW_OUT")) {
                            let stem = r.name.replace(['/', '\\', ':'], "_");
                            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}.{:x}.bin", p.offset)), code);
                        }
                        let mut off = 0usize;
                        let mut recent: Vec<usize> = Vec::new();
                        while off + 4 <= code.len() {
                            let Ok(d) = crate::gpu::rdna2::decode(&code[off..], off as u64) else { off += 4; continue };
                            let name = d.inst.name;
                            if name.eq_ignore_ascii_case("ds_swizzle_b32") {
                                let v = |n: &str| crate::gpu::rdna2::field_value(d.enc, &d.words, n).unwrap_or(0);
                                let mut m = swz.lock().unwrap();
                                m.entry(v("OFFSET0") | v("OFFSET1") << 8).or_insert((0, r.name.clone())).0 += 1;
                            }
                            if show && (name.eq_ignore_ascii_case("s_setpc_b64") || name.eq_ignore_ascii_case("s_swappc_b64")) {
                                let from = recent.get(recent.len().saturating_sub(12)).copied().unwrap_or(off);
                                let to = (off + 24).min(code.len());
                                let text: Vec<String> = crate::gpu::rdna2::disassemble(&code[from..to], from as u64, false)
                                    .iter()
                                    .map(|l| format!("  {:06x}: {}", l.pc, l.text))
                                    .collect();
                                println!("{} @{off:#x}:
{}", r.name, text.join("
"));
                            }
                            recent.push(off);
                            off += d.len;
                        }
                    }
                }
            });
        }
    });
    println!("ds_swizzle_b32 offsets (uses, example):");
    for (o, (n, ex)) in swz.into_inner().unwrap() {
        println!("  {o:#06x} {n:>6}  {ex}");
    }
}

/// `ds_swizzle_b32`'s four modes against the RDNA ISA (its FFT examples, rotate, quad
/// permute, and the bit mode with xor ≥ 16, which retail compute shaders use to swap the
/// 16-lane halves: 0x401F).
#[test]
fn ds_swizzle_modes_follow_the_isa() {
    use super::emit::swizzle_lane;
    let lanes = |off: u64| (0..32).map(|i| swizzle_lane(off, i)).collect::<Vec<u32>>();
    // ISA examples (inputs 1..0x20, so each value − 1 is the source lane)
    let fft0 = [0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17, 0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c, 0x08, 0x18, 0x10, 0x20];
    let fft10 = [0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c, 0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e, 0x14, 0x1c, 0x18, 0x20];
    assert_eq!(lanes(0xE000), fft0.iter().map(|v| v - 1).collect::<Vec<u32>>());
    assert_eq!(lanes(0xE010), fft10.iter().map(|v| v - 1).collect::<Vec<u32>>());
    assert_eq!(lanes(0xE01F), (0..32).collect::<Vec<u32>>(), "FFT mask 0x1F: no swizzle");
    // rotate left by 3 / right by 5 (offset[10]), whole 32 lanes
    assert_eq!(lanes(0xC000 | 3 << 5), (0..32).map(|i| (i + 3) % 32).collect::<Vec<u32>>());
    assert_eq!(lanes(0xC400 | 5 << 5), (0..32).map(|i| (i + 27) % 32).collect::<Vec<u32>>());
    // quad permute [3, 2, 1, 0]
    assert_eq!(lanes(0x8000 | 0x1B), (0..32).map(|i| (i & !3) | (3 - (i & 3))).collect::<Vec<u32>>());
    // bit mode: and 0x1F, xor 16 (swap halves) / xor 18
    assert_eq!(lanes(0x401F), (0..32).map(|i| i ^ 16).collect::<Vec<u32>>());
    assert_eq!(lanes(0x481F), (0..32).map(|i| i ^ 18).collect::<Vec<u32>>());
    assert_eq!(lanes(0x041F), (0..32).map(|i| i ^ 1).collect::<Vec<u32>>());
}

#[test]
fn gpu_ds_swizzle_modes() {
    if !gpu_ready() {
        return;
    }
    let offsets = [0x401Fu64, 0x481F, 0xE000, 0xE010, 0xC000 | 3 << 5, 0xC400 | 5 << 5 | 0x3, 0x8000 | 0x1B];
    let list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = offsets
        .iter()
        .enumerate()
        .map(|(k, &o)| ("DS_SWIZZLE_B32", "ENC_DS", vec![("VDST", 16 + k as u64), ("ADDR", 0), ("OFFSET0", o & 0xFF), ("OFFSET1", o >> 8)], None))
        .collect();
    let list: Vec<(&str, &str, &[(&str, u64)], Option<u32>)> = list.iter().map(|(n, e, f, l)| (*n, *e, f.as_slice(), *l)).collect();
    let code = prog(&list);
    let t = super::translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    let mut st = KernelState::default();
    for (l, r) in st.vgpr.iter_mut().enumerate() {
        r[0] = 100 + l as u32;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64u32 {
        for (k, &o) in offsets.iter().enumerate() {
            let want = 100 + ((l & 0x20) | super::emit::swizzle_lane(o, l));
            assert_eq!(r.vgpr[l as usize][16 + k], want, "lane {l} offset {o:#06x}");
        }
    }
}

/// The compiler's far jump (`s_getpc_b64`, add of a 64-bit constant, `s_setpc_b64`) out of
/// `s_branch` range is a direct branch: it gets its target, the program stays structured and
/// complete, and the code it jumps over does not run.
#[test]
fn far_jumps_are_direct_branches() {
    let far = |sub: bool, offset: u32| -> Vec<u8> {
        let (lo, hi) = if sub { ("S_SUB_U32", "S_SUBB_U32") } else { ("S_ADD_U32", "S_ADDC_U32") };
        prog(&[
            ("S_GETPC_B64", "ENC_SOP1", &[("SDST", 4)], None),
            (lo, "ENC_SOP2", &[("SDST", 4), ("SSRC0", 4), ("SSRC1", 255)], Some(offset)),
            (hi, "ENC_SOP2", &[("SDST", 5), ("SSRC0", 5), ("SSRC1", 128)], None),
            ("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 4)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)], None), // 0x14: v1 = 7 (jumped over)
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 137)], None), // 0x18: v2 = 9
        ])
    };
    let decoded = |code: &[u8]| {
        let mut v = Vec::new();
        let mut off = 0;
        while off + 4 <= code.len() {
            let d = crate::gpu::rdna2::decode(&code[off..], off as u64).unwrap();
            off += d.len;
            v.push(d);
        }
        crate::gpu::rdna2::resolve_far_jumps(&mut v);
        v
    };
    // s_getpc_b64 returns 0x4 (the next instruction): 0x4 + 0x14 = 0x18
    let code = far(false, 0x14);
    let insts = decoded(&code);
    assert_eq!(insts[3].far_target, Some(0x18));
    assert_eq!(crate::gpu::rdna2::branch_target(&insts[3]), Some(0x18));
    // backward form: 0x4 − 4 = 0x0
    assert_eq!(decoded(&far(true, 4))[3].far_target, Some(0));
    // an s_setpc_b64 of another register pair stays indirect
    let mut other = code.clone();
    other[0x10..0x14].copy_from_slice(&encode("S_SETPC_B64", "ENC_SOP1", &[("SSRC0", 6)], None).unwrap());
    assert_eq!(decoded(&other)[3].far_target, None);
    let t = super::translate(&other, &Options::default()).unwrap();
    assert!(!t.complete());

    let t = super::translate(&code, &Options::default()).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported);
    assert!(!t.hlsl.contains("pc =="), "far jump left the structured form:\n{}", t.hlsl);
    if !gpu_ready() {
        return;
    }
    let mut st = KernelState::default();
    for r in st.vgpr.iter_mut() {
        r[1] = 1;
        r[2] = 2;
    }
    let r = runner::run_code(&code, &st, &[]).unwrap();
    for l in 0..64 {
        assert_eq!((r.vgpr[l][1], r.vgpr[l][2]), (1, 9), "lane {l}");
    }
}

#[test]
fn mesh_wave32_emits_wave32_logical_wave64() {
    let code = 0xBF81_0000u32.to_le_bytes(); // s_endpgm
    let ms = super::MsInterface { threads: 128, max_vertices: 64, max_primitives: 64, outputs: vec![] };
    let native = translate(&code, &Options { mode: Mode::Mesh(ms.clone()), wave64: true, mesh_wave32: false, ..Default::default() }).unwrap();
    assert!(native.hlsl.contains("[WaveSize(64)]"));
    assert!(native.hlsl.contains("[numthreads(128, 1, 1)]"));
    assert!(!native.hlsl.contains("xcl_logical_wave64 = true"));

    let split = translate(&code, &Options { mode: Mode::Mesh(ms), wave64: true, mesh_wave32: true, ..Default::default() }).unwrap();
    assert!(split.hlsl.contains("[WaveSize(32)]"));
    assert!(split.hlsl.contains("[numthreads(128, 1, 1)]"));
    assert!(split.hlsl.contains("xcl_half = (gi >> 5) & 1u;"));
    assert!(split.hlsl.contains("xcl_logical_wave64 = true;"));
    assert!(split.hlsl.contains("lane_id = WaveGetLaneIndex() + 32u * xcl_half;"));
    assert!(split.complete(), "{:?}", split.unsupported_why);
}

#[test]
fn mesh_wave32_rejects_cross_half_operations() {
    let ms = super::MsInterface { threads: 128, max_vertices: 64, max_primitives: 64, outputs: vec![] };
    let lane = prog(&[("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 0), ("SRC0", V)], None)]);
    let branch = prog(&[("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 0)], None)]);
    for (code, hazard) in [(&lane, "cross-half lane exchange"), (&branch, "branch on a whole-wave mask")] {
        let e = super::translate_mesh(code, ms.clone(), true, true).err().expect("split mesh must refuse");
        assert!(e.contains("mesh program requires unsupported cross-half wave64 semantics") && e.contains(hazard), "{e}");
        // native wave64 and wave32 programs are not split, so nothing is refused
        assert!(super::translate_mesh(code, ms.clone(), true, false).is_ok());
        assert!(super::translate_mesh(code, ms.clone(), false, true).is_ok());
    }

    let plain = prog(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", V)], None)]);
    let split = super::translate_mesh(&plain, ms, true, true).unwrap();
    assert!(split.hlsl.contains("xcl_logical_wave64 = true;"));
    assert!(split.complete(), "{:?}", split.unsupported_why);
}
