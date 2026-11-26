//! The FPv5-D16 data-processing instructions of the Cortex-M7. Encodings are
//! those GNU as 2.45.1 assembles (`-mcpu=cortex-m7 -mfpu=fpv5-d16
//! -mthumb`), noted next to each instruction; expected results follow the
//! Armv7-M ARM (DDI 0403E.e) A7.7 and A2.5 pseudocode and IEEE 754, compared
//! as raw bits with the FPSCR.

use crate::isa::arm::aarch32::vfp::Fpscr;
use crate::isa::arm::common::memory::{ArmMemory, FlatMemory};
use crate::isa::arm::cortex_m::cpu::CortexMCpu;
use crate::isa::arm::cortex_m::exec::Fault;
use crate::isa::arm::cortex_m::scb::CortexMVariant;

const CODE: u32 = 0x100;
const STACK: u32 = 0x8000;
/// CPACR with full access to CP10 and CP11.
const CPACR_FULL: u32 = 0x00F0_0000;

/// FPSCR cumulative exception flags: Invalid Operation, Division by Zero,
/// Overflow, Inexact, Input Denormal.
const IOC: u32 = 1 << 0;
const DZC: u32 = 1 << 1;
const OFC: u32 = 1 << 2;
const IXC: u32 = 1 << 4;
const IDC: u32 = 1 << 7;
const FLAGS: u32 = 0x9F;
/// FPSCR controls: Default NaN, Flush-to-zero, and RMode (bits 23:22).
const DN: u32 = 1 << 25;
const FZ: u32 = 1 << 24;
const RP: u32 = 0b01 << 22;
const RM: u32 = 0b10 << 22;
const RZ: u32 = 0b11 << 22;

/// APSR flags in xPSR.
const N: u32 = 1 << 31;
const Z: u32 = 1 << 30;
const V: u32 = 1 << 28;

const ONE: u32 = 0x3F80_0000;
const TWO: u32 = 0x4000_0000;
const THREE: u32 = 0x4040_0000;
const QNAN: u32 = 0x7FC0_0001;
const SNAN: u32 = 0x7F80_0001;
const DEFAULT_NAN: u32 = 0x7FC0_0000;

fn m7(code: &[u8]) -> CortexMCpu {
    let mut memory = FlatMemory::new(0, 0x1_0000);
    memory.write(u64::from(CODE), code).unwrap();
    let mut cpu = CortexMCpu::new(CortexMVariant::CortexM7, Box::new(memory));
    cpu.set_floating_point(true);
    cpu.ppb.cpacr = CPACR_FULL;
    cpu.pc = CODE;
    cpu.sp_main = STACK;
    cpu
}

/// A 32-bit Thumb instruction in memory: its first halfword, then its
/// second, each little-endian.
fn thumb32(hw1: u16, hw2: u16) -> [u8; 4] {
    let (a, b) = (hw1.to_le_bytes(), hw2.to_le_bytes());
    [a[0], a[1], b[0], b[1]]
}

/// Runs the instruction `hw1 hw2` once on a core `setup` prepares.
fn run(hw1: u16, hw2: u16, setup: impl FnOnce(&mut CortexMCpu)) -> CortexMCpu {
    let mut cpu = m7(&thumb32(hw1, hw2));
    setup(&mut cpu);
    if let Err(fault) = cpu.execute_one() {
        panic!("{hw1:04X} {hw2:04X}: {fault:?}");
    }
    cpu
}

fn fault(hw1: u16, hw2: u16) -> Option<Fault> {
    m7(&thumb32(hw1, hw2)).execute_one().err()
}

/// The FPSCR of an active FP context (CONTROL.FPCA set): the first FP
/// instruction keeps it instead of taking FPDSCR's defaults
/// (`ExecuteFPCheck`, B1.5.7).
fn set_fpscr(cpu: &mut CortexMCpu, value: u32) {
    cpu.control |= 1 << 2;
    cpu.set_fpscr(Fpscr::from_bits(value));
}

fn flags(cpu: &CortexMCpu) -> u32 {
    cpu.fpscr().bits() & FLAGS
}

/// A single-precision operation on s1 and s2 into s0, with the FPSCR
/// `controls` and s0 holding `accumulator`: s0 and the exception flags.
fn single(hw1: u16, hw2: u16, accumulator: u32, a: u32, b: u32, controls: u32) -> (u32, u32) {
    let cpu = run(hw1, hw2, |cpu| {
        set_fpscr(cpu, controls);
        cpu.set_s(0, accumulator);
        cpu.set_s(1, a);
        cpu.set_s(2, b);
    });
    (cpu.s(0), flags(&cpu))
}

/// A one-operand single-precision operation on s1 into s0.
fn single_unary(hw1: u16, hw2: u16, a: u32, controls: u32) -> (u32, u32) {
    single(hw1, hw2, 0, a, 0, controls)
}

/// A double-precision operation on d1 and d2 into d0.
fn double(hw1: u16, hw2: u16, accumulator: u64, a: u64, b: u64) -> (u64, u32) {
    let cpu = run(hw1, hw2, |cpu| {
        cpu.set_d(0, accumulator);
        cpu.set_d(1, a);
        cpu.set_d(2, b);
    });
    (cpu.d(0), flags(&cpu))
}

#[test]
fn three_register_arithmetic_in_both_precisions() {
    let (a, b) = (0x3FC0_0000, 0x4010_0000); // 1.5, 2.25
    // vadd.f32 s0, s1, s2 (EE30 0A81) ; vsub (EE30 0AC1) ; vmul (EE20 0A81)
    assert_eq!(single(0xEE30, 0x0A81, 0, a, b, 0), (0x4070_0000, 0));
    assert_eq!(single(0xEE30, 0x0AC1, 0, a, b, 0), (0xBF40_0000, 0));
    assert_eq!(single(0xEE20, 0x0A81, 0, a, b, 0), (0x4058_0000, 0));
    // vdiv.f32 (EE80 0A81): 2/3 rounds up to ...AB
    assert_eq!(single(0xEE80, 0x0A81, 0, a, b, 0), (0x3F2A_AAAB, IXC));

    let (a, b) = (0x3FF8_0000_0000_0000, 0x4002_0000_0000_0000);
    // vadd.f64 d0, d1, d2 (EE31 0B02) ; vdiv.f64 (EE81 0B02): 2/3 rounds
    // down to ...55
    assert_eq!(double(0xEE31, 0x0B02, 0, a, b), (0x400E_0000_0000_0000, 0));
    assert_eq!(
        double(0xEE81, 0x0B02, 0, a, b),
        (0x3FE5_5555_5555_5555, IXC)
    );
    // vsub.f64 (EE31 0B42) ; vmul.f64 (EE21 0B02)
    assert_eq!(double(0xEE31, 0x0B42, 0, a, b), (0xBFE8_0000_0000_0000, 0));
    assert_eq!(double(0xEE21, 0x0B02, 0, a, b), (0x400B_0000_0000_0000, 0));
}

#[test]
fn high_registers_follow_the_register_numbering() {
    // vadd.f32 s31, s30, s29 (EE7F FA2E): Vd:D, Vn:N, Vm:M
    let cpu = run(0xEE7F, 0xFA2E, |cpu| {
        cpu.set_s(30, ONE);
        cpu.set_s(29, TWO);
    });
    assert_eq!(cpu.s(31), THREE);
    // vadd.f64 d15, d14, d13 (EE3E FB0D): D:Vd, N:Vn, M:Vm
    let cpu = run(0xEE3E, 0xFB0D, |cpu| {
        cpu.set_d(14, 0x3FF0_0000_0000_0000);
        cpu.set_d(13, 0x4000_0000_0000_0000);
    });
    assert_eq!(cpu.d(15), 0x4008_0000_0000_0000);
}

#[test]
fn accumulating_forms_negate_as_specified() {
    // Sd = 1.0, Sn = 2.0, Sm = 3.0; every result is exact.
    let cases = [
        ("vmla", 0xEE00, 0x0A81, 0x40E0_0000),  // 1 + 6
        ("vmls", 0xEE00, 0x0AC1, 0xC0A0_0000),  // 1 - 6
        ("vnmla", 0xEE10, 0x0AC1, 0xC0E0_0000), // -1 - 6
        ("vnmls", 0xEE10, 0x0A81, 0x40A0_0000), // -1 + 6
        ("vnmul", 0xEE20, 0x0AC1, 0xC0C0_0000), // -(2 * 3)
        ("vfma", 0xEEA0, 0x0A81, 0x40E0_0000),  // 1 + 6
        ("vfms", 0xEEA0, 0x0AC1, 0xC0A0_0000),  // 1 + (-2) * 3
        ("vfnma", 0xEE90, 0x0AC1, 0xC0E0_0000), // -1 + (-2) * 3
        ("vfnms", 0xEE90, 0x0A81, 0x40A0_0000), // -1 + 2 * 3
    ];
    for (name, hw1, hw2, expected) in cases {
        assert_eq!(
            single(hw1, hw2, ONE, TWO, THREE, 0),
            (expected, 0),
            "{name}"
        );
    }
    let (one, two, three) = (
        0x3FF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x4008_0000_0000_0000,
    );
    // vmla.f64 d0, d1, d2 (EE01 0B02) ; vfma.f64 (EEA1 0B02)
    assert_eq!(
        double(0xEE01, 0x0B02, one, two, three),
        (0x401C_0000_0000_0000, 0)
    );
    assert_eq!(
        double(0xEEA1, 0x0B02, one, two, three),
        (0x401C_0000_0000_0000, 0)
    );
}

#[test]
fn fused_multiply_add_rounds_once() {
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24: the product alone rounds (ties to
    // even) to 1 + 2^-11, so VMLA adding -(1 + 2^-11) gives +0 and Inexact;
    // VFMA keeps 2^-24 exactly.
    let (a, minus) = (0x3F80_0800, 0xBF80_1000);
    // vmla.f32 s0, s1, s2 (EE00 0A81) ; vfma.f32 (EEA0 0A81)
    assert_eq!(single(0xEE00, 0x0A81, minus, a, a, 0), (0, IXC));
    assert_eq!(single(0xEEA0, 0x0A81, minus, a, a, 0), (0x3380_0000, 0));
}

#[test]
fn special_values_raise_their_exceptions() {
    // vdiv.f32 s0, s1, s2 (EE80 0A81)
    assert_eq!(
        single(0xEE80, 0x0A81, 0, ONE, 0, 0),
        (0x7F80_0000, DZC),
        "1 / 0"
    );
    assert_eq!(
        single(0xEE80, 0x0A81, 0, 0, 0, 0),
        (DEFAULT_NAN, IOC),
        "0 / 0"
    );
    // vadd.f32 s0, s1, s2 (EE30 0A81): a quiet NaN propagates, a signaling
    // NaN is quieted with Invalid Operation, DN gives the default NaN
    assert_eq!(single(0xEE30, 0x0A81, 0, QNAN, ONE, 0), (QNAN, 0));
    assert_eq!(single(0xEE30, 0x0A81, 0, SNAN, ONE, 0), (QNAN, IOC));
    assert_eq!(single(0xEE30, 0x0A81, 0, SNAN, ONE, DN), (DEFAULT_NAN, IOC));
    // vmul.f32 (EE20 0A81): the largest single times 2 overflows
    assert_eq!(
        single(0xEE20, 0x0A81, 0, 0x7F7F_FFFF, TWO, 0),
        (0x7F80_0000, OFC | IXC)
    );
    // FZ flushes a denormal input to zero, with Input Denormal
    assert_eq!(single(0xEE30, 0x0A81, 0, 1, 0, 0), (1, 0));
    assert_eq!(single(0xEE30, 0x0A81, 0, 1, 0, FZ), (0, IDC));
    // vsqrt.f32 s0, s1 (EEB1 0AE0)
    assert_eq!(single_unary(0xEEB1, 0x0AE0, TWO, 0), (0x3FB5_04F3, IXC));
    assert_eq!(
        single_unary(0xEEB1, 0x0AE0, 0xBF80_0000, 0),
        (DEFAULT_NAN, IOC)
    );
    // vabs.f32 s0, s1 (EEB0 0AE0) ; vneg.f32 (EEB1 0A60): the sign bit
    // only, a NaN unprocessed
    assert_eq!(single_unary(0xEEB0, 0x0AE0, 0xFF80_0001, 0), (SNAN, 0));
    assert_eq!(single_unary(0xEEB1, 0x0A60, 0xFF80_0001, 0), (SNAN, 0));
    // vabs.f64 d0, d1 (EEB0 0BC1) ; vneg.f64 (EEB1 0B41) ; vsqrt.f64 (EEB1 0BC1)
    let minus_four = 0xC010_0000_0000_0000;
    assert_eq!(
        double(0xEEB0, 0x0BC1, 0, minus_four, 0),
        (0x4010_0000_0000_0000, 0)
    );
    assert_eq!(
        double(0xEEB1, 0x0B41, 0, minus_four, 0),
        (0x4010_0000_0000_0000, 0)
    );
    assert_eq!(
        double(0xEEB1, 0x0BC1, 0, 0x4010_0000_0000_0000, 0),
        (0x4000_0000_0000_0000, 0)
    );
}

#[test]
fn rounding_follows_fpscr_rmode() {
    // vdiv.f32 s0, s1, s2 (EE80 0A81): 1/3 = 1.0101...b x 2^-2, the 24th
    // bit followed by 1010...: up to ...AB to nearest, down to ...AA towards
    // zero or minus infinity; -1/3 mirrors it.
    let minus_one = 0xBF80_0000;
    for (rmode, positive, negative) in [
        (0, 0x3EAA_AAAB, 0xBEAA_AAAB),
        (RP, 0x3EAA_AAAB, 0xBEAA_AAAA),
        (RM, 0x3EAA_AAAA, 0xBEAA_AAAB),
        (RZ, 0x3EAA_AAAA, 0xBEAA_AAAA),
    ] {
        assert_eq!(
            single(0xEE80, 0x0A81, 0, ONE, THREE, rmode),
            (positive, IXC)
        );
        assert_eq!(
            single(0xEE80, 0x0A81, 0, minus_one, THREE, rmode),
            (negative, IXC)
        );
    }

    // the first FP instruction of a context (CONTROL.FPCA clear, FPCCR.ASPEN
    // set at reset) takes RMode from FPDSCR
    let cpu = run(0xEE80, 0x0A81, |cpu| {
        cpu.ppb.fpdscr = RZ;
        cpu.set_s(1, ONE);
        cpu.set_s(2, THREE);
    });
    assert_eq!(cpu.s(0), 0x3EAA_AAAA);
    assert_eq!(cpu.fpscr().bits() & (0b11 << 22), RZ);
}

#[test]
fn compare_sets_the_fpscr_flags() {
    /// vcmp.f32 s0, s1 (EEB4 0A60) or vcmpe.f32 (EEB4 0AE0): NZCV and the
    /// exception flags.
    fn compare(hw2: u16, a: u32, b: u32) -> (u32, u32) {
        let cpu = run(0xEEB4, hw2, |cpu| {
            cpu.set_s(0, a);
            cpu.set_s(1, b);
        });
        (cpu.fpscr().bits() >> 28, flags(&cpu))
    }
    assert_eq!(compare(0x0A60, ONE, TWO), (0b1000, 0), "less");
    assert_eq!(compare(0x0A60, TWO, TWO), (0b0110, 0), "equal");
    assert_eq!(compare(0x0A60, THREE, TWO), (0b0010, 0), "greater");
    assert_eq!(compare(0x0A60, QNAN, TWO), (0b0011, 0), "unordered, quiet");
    assert_eq!(
        compare(0x0AE0, QNAN, TWO),
        (0b0011, IOC),
        "VCMPE on a quiet NaN"
    );
    assert_eq!(
        compare(0x0A60, SNAN, TWO),
        (0b0011, IOC),
        "VCMP on a signaling NaN"
    );

    // vcmp.f32 s0, #0.0 (EEB5 0A40): -0.0 equals +0.0
    let cpu = run(0xEEB5, 0x0A40, |cpu| cpu.set_s(0, 0x8000_0000));
    assert_eq!(cpu.fpscr().bits() >> 28, 0b0110);
    // vcmpe.f64 d0, #0.0 (EEB5 0BC0) ; vcmp.f64 d0, d1 (EEB4 0B41)
    let cpu = run(0xEEB5, 0x0BC0, |cpu| cpu.set_d(0, 0xBFF0_0000_0000_0000));
    assert_eq!(cpu.fpscr().bits() >> 28, 0b1000);
    let cpu = run(0xEEB4, 0x0B41, |cpu| {
        cpu.set_d(0, 0x4008_0000_0000_0000);
        cpu.set_d(1, 0x3FF0_0000_0000_0000);
    });
    assert_eq!(cpu.fpscr().bits() >> 28, 0b0010);

    // vcmp.f32 s0, s1 ; vmrs APSR_nzcv, fpscr (EEF1 FA10)
    let mut code = thumb32(0xEEB4, 0x0A60).to_vec();
    code.extend(thumb32(0xEEF1, 0xFA10));
    let mut cpu = m7(&code);
    cpu.set_s(0, ONE);
    cpu.set_s(1, TWO);
    cpu.execute_one().unwrap();
    cpu.execute_one().unwrap();
    assert_eq!(cpu.xpsr >> 28, 0b1000);
}

#[test]
fn conversions_with_integers() {
    let minus_two_and_a_half = 0xC020_0000;
    // vcvt.s32.f32 s0, s1 (EEBD 0AE0): towards zero
    assert_eq!(
        single_unary(0xEEBD, 0x0AE0, minus_two_and_a_half, 0),
        (-2i32 as u32, IXC)
    );
    // vcvtr.s32.f32 s0, s1 (EEBD 0A60): the FPSCR's rounding
    for (rmode, expected) in [(0, -2), (RP, -2), (RM, -3), (RZ, -2)] {
        assert_eq!(
            single_unary(0xEEBD, 0x0A60, minus_two_and_a_half, rmode),
            (expected as u32, IXC),
            "rmode {rmode:#x}"
        );
    }
    // out of range: saturated, Invalid Operation only (FPToFixed)
    assert_eq!(
        single_unary(0xEEBD, 0x0AE0, 0x4F32_D05E, 0),
        (0x7FFF_FFFF, IOC),
        "3e9 to s32"
    );
    // vcvt.u32.f32 s0, s1 (EEBC 0AE0)
    assert_eq!(
        single_unary(0xEEBC, 0x0AE0, 0xBF80_0000, 0),
        (0, IOC),
        "-1 to u32"
    );
    // vcvtr.s32.f64 s0, d1 (EEBD 0B41): 2.5 to nearest, ties to even
    let cpu = run(0xEEBD, 0x0B41, |cpu| cpu.set_d(1, 0x4004_0000_0000_0000));
    assert_eq!((cpu.s(0), flags(&cpu)), (2, IXC));
    // vcvt.f32.s32 s0, s1 (EEB8 0AE0): 2^24 + 1 rounds to 2^24
    assert_eq!(
        single_unary(0xEEB8, 0x0AE0, (1 << 24) + 1, 0),
        (0x4B80_0000, IXC)
    );
    // vcvt.f32.u32 s0, s1 (EEB8 0A60): 2^32 - 1 rounds to 2^32
    assert_eq!(
        single_unary(0xEEB8, 0x0A60, u32::MAX, 0),
        (0x4F80_0000, IXC)
    );
    // vcvt.f64.u32 d0, s2 (EEB8 0B41) ; vcvt.f64.s32 d0, s2 (EEB8 0BC1): exact
    let cpu = run(0xEEB8, 0x0B41, |cpu| cpu.set_s(2, u32::MAX));
    assert_eq!((cpu.d(0), flags(&cpu)), (0x41EF_FFFF_FFE0_0000, 0));
    let cpu = run(0xEEB8, 0x0BC1, |cpu| cpu.set_s(2, 0x8000_0000));
    assert_eq!((cpu.d(0), flags(&cpu)), (0xC1E0_0000_0000_0000, 0));
}

#[test]
fn conversions_with_fixed_point() {
    /// A conversion of s0 in place.
    fn in_place(hw1: u16, hw2: u16, value: u32) -> (u32, u32) {
        let cpu = run(hw1, hw2, |cpu| cpu.set_s(0, value));
        (cpu.s(0), flags(&cpu))
    }
    // vcvt.s16.f32 s0, s0, #8 (EEBE 0A44): sign-extended to 32 bits
    assert_eq!(in_place(0xEEBE, 0x0A44, 0x3FC0_0000), (0x0000_0180, 0));
    assert_eq!(in_place(0xEEBE, 0x0A44, 0xBFC0_0000), (0xFFFF_FE80, 0));
    // 1 + 2^-9 is 256.5 sixteenths of 2^-8: towards zero, Inexact
    assert_eq!(in_place(0xEEBE, 0x0A44, 0x3F80_4000), (0x0000_0100, IXC));
    // vcvt.u16.f32 s0, s0, #8 (EEBF 0A44): 300 x 256 saturates
    assert_eq!(in_place(0xEEBF, 0x0A44, 0x4396_0000), (0x0000_FFFF, IOC));
    // vcvt.f32.s16 s0, s0, #8 (EEBA 0A44): the low halfword only
    assert_eq!(in_place(0xEEBA, 0x0A44, 0x1234_FF80), (0xBF00_0000, 0));
    // vcvt.s32.f64 d0, d0, #16 (EEBE 0BC8): sign-extended to 64 bits
    let cpu = run(0xEEBE, 0x0BC8, |cpu| cpu.set_d(0, 0xBFF8_0000_0000_0000));
    assert_eq!(cpu.d(0), 0xFFFF_FFFF_FFFE_8000);
    // vcvt.f64.u32 d0, d0, #16 (EEBB 0BC8): the low word only
    let cpu = run(0xEEBB, 0x0BC8, |cpu| cpu.set_d(0, 0xDEAD_BEEF_0001_8000));
    assert_eq!(cpu.d(0), 0x3FF8_0000_0000_0000);
}

#[test]
fn conversions_between_precisions() {
    // vcvt.f64.f32 d0, s2 (EEB7 0AC1): 0.1f exactly
    let cpu = run(0xEEB7, 0x0AC1, |cpu| cpu.set_s(2, 0x3DCC_CCCD));
    assert_eq!((cpu.d(0), flags(&cpu)), (0x3FB9_9999_A000_0000, 0));
    // vcvt.f32.f64 s0, d1 (EEB7 0BC1): 0.1 rounds
    let cpu = run(0xEEB7, 0x0BC1, |cpu| cpu.set_d(1, 0x3FB9_9999_9999_999A));
    assert_eq!((cpu.s(0), flags(&cpu)), (0x3DCC_CCCD, IXC));

    // vcvtt.f32.f16 s0, s1 (EEB2 0AE0) ; vcvtb.f32.f16 (EEB2 0A60)
    assert_eq!(single_unary(0xEEB2, 0x0AE0, 0x3C00_C000, 0), (ONE, 0));
    assert_eq!(
        single_unary(0xEEB2, 0x0A60, 0x3C00_C000, 0),
        (0xC000_0000, 0)
    );
    // vcvtb.f16.f32 s0, s1 (EEB3 0A60) ; vcvtt.f16.f32 (EEB3 0AE0): the
    // other halfword of s0 is kept; 65504 is the largest half
    assert_eq!(
        single(0xEEB3, 0x0A60, 0xAAAA_5555, ONE, 0, 0),
        (0xAAAA_3C00, 0)
    );
    assert_eq!(
        single(0xEEB3, 0x0AE0, 0xAAAA_5555, 0x477F_E000, 0, 0),
        (0x7BFF_5555, 0)
    );
    // vcvtb.f64.f16 d0, s2 (EEB2 0B41) ; vcvtt.f16.f64 s0, d1 (EEB3 0BC1)
    let cpu = run(0xEEB2, 0x0B41, |cpu| cpu.set_s(2, 0x0000_3800));
    assert_eq!(cpu.d(0), 0x3FE0_0000_0000_0000);
    let cpu = run(0xEEB3, 0x0BC1, |cpu| {
        cpu.set_s(0, 0x0000_1234);
        cpu.set_d(1, 0x3FF0_0000_0000_0000);
    });
    assert_eq!(cpu.s(0), 0x3C00_1234);
}

#[test]
fn round_to_integral() {
    let two_and_a_half = 0x4020_0000;
    // vrinta, vrintn, vrintp, vrintm .f32 s0, s1 (FEB8..FEBB 0A60): no
    // Inexact
    for (name, hw1, expected) in [
        ("vrinta", 0xFEB8, THREE),
        ("vrintn", 0xFEB9, TWO),
        ("vrintp", 0xFEBA, THREE),
        ("vrintm", 0xFEBB, TWO),
    ] {
        assert_eq!(
            single_unary(hw1, 0x0A60, two_and_a_half, 0),
            (expected, 0),
            "{name}"
        );
    }
    // vrintz.f32 s0, s1 (EEB6 0AE0) ; vrintr (EEB6 0A60) by the FPSCR
    assert_eq!(
        single_unary(0xEEB6, 0x0AE0, 0xC020_0000, 0),
        (0xC000_0000, 0)
    );
    assert_eq!(
        single_unary(0xEEB6, 0x0A60, 0xC020_0000, RM),
        (0xC040_0000, 0)
    );
    // vrintx.f32 s0, s1 (EEB7 0A60): Inexact
    assert_eq!(single_unary(0xEEB7, 0x0A60, two_and_a_half, 0), (TWO, IXC));
    // vrinta.f64 d0, d1 (FEB8 0B41): -0.5 ties away to -1
    assert_eq!(
        double(0xFEB8, 0x0B41, 0, 0xBFE0_0000_0000_0000, 0),
        (0xBFF0_0000_0000_0000, 0)
    );
}

#[test]
fn conversions_with_directed_rounding() {
    // vcvta, vcvtn, vcvtp, vcvtm .s32.f32 s0, s1 (FEBC..FEBF 0AE0)
    for (name, hw1, positive, negative) in [
        ("vcvta", 0xFEBC, 3, -3),
        ("vcvtn", 0xFEBD, 2, -2),
        ("vcvtp", 0xFEBE, 3, -2),
        ("vcvtm", 0xFEBF, 2, -3),
    ] {
        assert_eq!(
            single_unary(hw1, 0x0AE0, 0x4020_0000, 0),
            (positive as u32, IXC),
            "{name} 2.5"
        );
        assert_eq!(
            single_unary(hw1, 0x0AE0, 0xC020_0000, 0),
            (negative as u32, IXC),
            "{name} -2.5"
        );
    }
    // vcvta.u32.f64 s0, d1 (FEBC 0B41)
    let cpu = run(0xFEBC, 0x0B41, |cpu| cpu.set_d(1, 0x4004_0000_0000_0000));
    assert_eq!(cpu.s(0), 3);
}

#[test]
fn maximum_and_minimum_numbers() {
    // vmaxnm.f32 s0, s1, s2 (FE80 0A81) ; vminnm.f32 (FE80 0AC1): a quiet
    // NaN loses to a number, a signaling NaN is quieted
    assert_eq!(single(0xFE80, 0x0A81, 0, ONE, DEFAULT_NAN, 0), (ONE, 0));
    assert_eq!(single(0xFE80, 0x0A81, 0, ONE, SNAN, 0), (QNAN, IOC));
    assert_eq!(single(0xFE80, 0x0A81, 0, 0x8000_0000, 0, 0), (0, 0));
    assert_eq!(
        single(0xFE80, 0x0AC1, 0, 0x8000_0000, 0, 0),
        (0x8000_0000, 0)
    );
    // vmaxnm.f64 d0, d1, d2 (FE81 0B02)
    assert_eq!(
        double(
            0xFE81,
            0x0B02,
            0,
            0x4000_0000_0000_0000,
            0x4008_0000_0000_0000
        ),
        (0x4008_0000_0000_0000, 0)
    );
}

#[test]
fn vsel_takes_its_condition_from_apsr() {
    /// VSEL into s0 from s1 = 1.0 or s2 = 2.0, with the APSR flags `apsr`.
    fn select(hw1: u16, apsr: u32) -> u32 {
        run(hw1, 0x0A81, |cpu| {
            cpu.xpsr = (cpu.xpsr & 0x0FFF_FFFF) | apsr;
            cpu.set_s(1, ONE);
            cpu.set_s(2, TWO);
        })
        .s(0)
    }
    // vseleq (FE00), vselvs (FE10), vselge (FE20) .f32 s0, s1, s2
    assert_eq!(select(0xFE00, Z), ONE);
    assert_eq!(select(0xFE00, 0), TWO);
    assert_eq!(select(0xFE10, V), ONE);
    assert_eq!(select(0xFE10, 0), TWO);
    assert_eq!(select(0xFE20, N | V), ONE);
    assert_eq!(select(0xFE20, N), TWO);
    // vselgt.f64 d0, d1, d2 (FE31 0B02): Z clear and N == V
    for (apsr, expected) in [(0, 1u64), (Z, 2), (N, 2)] {
        let cpu = run(0xFE31, 0x0B02, |cpu| {
            cpu.xpsr = (cpu.xpsr & 0x0FFF_FFFF) | apsr;
            cpu.set_d(1, 1);
            cpu.set_d(2, 2);
        });
        assert_eq!(cpu.d(0), expected, "APSR {apsr:#x}");
    }
}

#[test]
fn reserved_and_absent_encodings_are_undefined() {
    let undefined = |hw1, hw2| matches!(fault(hw1, hw2), Some(Fault::Undefined(_)));
    // vadd.f64 with D set: D16, beyond FPv5-D16
    assert!(undefined(0xEE71, 0x0B02));
    // vdiv.f32 with bit 6 set
    assert!(undefined(0xEE80, 0x0AC1));
    // opc1 = 1x11, opc2 = 1001
    assert!(undefined(0xEEB9, 0x0A40));
    // vcmp.f32 s0, #0.0 with a nonzero (0) bit
    assert!(undefined(0xEEB5, 0x0A41));
    // vcvt.s16.f32 with imm4:i = 17 > 16
    assert!(undefined(0xEEBE, 0x0A68));
    // vsel with bit 6 set
    assert!(undefined(0xFE00, 0x0AC1));
    // a T = 1 load/store form of CP10
    assert!(undefined(0xFD80, 0x0A00));

    // it eq (BF08), then vrinta.f32 s0, s1 (FEB8 0A60): UNPREDICTABLE in an
    // IT block
    let mut code = vec![0x08, 0xBF];
    code.extend(thumb32(0xFEB8, 0x0A60));
    let mut cpu = m7(&code);
    cpu.set_z(true);
    cpu.execute_one().unwrap();
    assert!(matches!(cpu.execute_one(), Err(Fault::Undefined(_))));

    assert!(fault(0xEE30, 0x0A81).is_none());
}
