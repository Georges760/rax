//! Floating-point data processing (Armv7-M ARM, DDI 0403E.e, A6.4, Table
//! A6-5): the FPv5-D16 instructions of the Cortex-M7.
//!
//! - Three registers: VMLA, VMLS, VNMLA, VNMLS, VNMUL, VMUL, VADD, VSUB,
//!   VDIV, VFMA, VFMS, VFNMA, VFNMS.
//! - Two registers: VMOV (immediate, register), VABS, VNEG, VSQRT, VCMP,
//!   VCMPE, VRINTZ, VRINTR, VRINTX.
//! - Conversions: VCVTB and VCVTT (half precision), VCVT between single and
//!   double precision, with 32-bit integers (and VCVTR), with fixed point.
//! - The FPv5 additions, 0b1111 in bits 31:28: VSEL, VMAXNM, VMINNM, VRINTA,
//!   VRINTN, VRINTP, VRINTM, VCVTA, VCVTN, VCVTP, VCVTM. They are
//!   UNPREDICTABLE in an IT block, taken as UNDEFINED here.
//!
//! The arithmetic is that of the AArch32 VFP (`aarch32::vfp`): IEEE 754
//! results under the FPSCR's RMode, FZ, DN and AHP, with the cumulative
//! exception flags. The M profile has no trapped floating-point exceptions
//! and no half-precision arithmetic, only conversions. A double register
//! beyond D15 is UNDEFINED.

use super::Fault;
use super::fp::D_REGS;
use crate::isa::arm::aarch32::vfp::{self, Fpscr, RoundingMode};
use crate::isa::arm::common::cpu::CpuExit;
use crate::isa::arm::cortex_m::cpu::CortexMCpu;
use crate::smir::interpret::SmirInterpreter;

/// The operations on two operands and, for the accumulating ones, the
/// destination.
#[derive(Clone, Copy, Debug)]
enum Arithmetic {
    Vmla,
    Vmls,
    Vnmla,
    Vnmls,
    Vnmul,
    Vmul,
    Vadd,
    Vsub,
    Vdiv,
    Vfma,
    Vfms,
    Vfnma,
    Vfnms,
    Vmaxnm,
    Vminnm,
}

/// The operations on one operand of the destination's precision.
#[derive(Clone, Copy, Debug)]
enum Unary {
    Vabs,
    Vneg,
    Vsqrt,
    /// VRINT: the rounding mode, and whether an inexact result raises
    /// Inexact.
    Vrint(RoundingMode, bool),
}

/// The registers of an encoding whose operands share one precision:
/// `D:Vd`, `N:Vn`, `M:Vm` for doubles, `Vd:D`, `Vn:N`, `Vm:M` for singles.
#[derive(Clone, Copy)]
struct Registers {
    double: bool,
    d: u32,
    n: u32,
    m: u32,
}

impl Registers {
    fn of(insn: u32) -> Self {
        let double = insn & (1 << 8) != 0;
        Self {
            double,
            d: destination(insn, double),
            n: CortexMCpu::fp_register(double, (insn >> 16) & 0xF, (insn >> 7) & 1),
            m: operand(insn, double),
        }
    }

    /// Whether every register this encoding names exists: always for
    /// singles, D0 to D15 for doubles.
    fn exist(&self, registers: &[u32]) -> bool {
        !self.double || registers.iter().all(|&r| r < D_REGS)
    }
}

/// The destination register, `D:Vd` (double) or `Vd:D` (single).
fn destination(insn: u32, double: bool) -> u32 {
    CortexMCpu::fp_register(double, (insn >> 12) & 0xF, (insn >> 22) & 1)
}

/// The last operand register, `M:Vm` (double) or `Vm:M` (single).
fn operand(insn: u32, double: bool) -> u32 {
    CortexMCpu::fp_register(double, insn & 0xF, (insn >> 5) & 1)
}

/// The rounding mode of the directed FPv5 additions, RM in bits 17:16:
/// ties away, ties to even, towards plus infinity, towards minus infinity.
fn directed_rounding(insn: u32) -> RoundingMode {
    match (insn >> 16) & 3 {
        0b00 => RoundingMode::RoundTiesAway,
        0b01 => RoundingMode::RoundNearest,
        0b10 => RoundingMode::RoundPlusInf,
        _ => RoundingMode::RoundMinusInf,
    }
}

/// `word` with `half` in its halfword at `shift` (0 or 16).
fn insert_half(word: u32, half: u16, shift: u32) -> u32 {
    (word & !(0xFFFF << shift)) | (u32::from(half) << shift)
}

/// The `size`-bit result of FPToFixed in 64 bits: zero-extended for an
/// unsigned result, sign-extended for a signed one.
fn extend_fixed(result: u64, size: u32, unsigned: bool) -> u64 {
    let low = result & (u64::MAX >> (64 - size));
    if unsigned {
        low
    } else {
        ((low << (64 - size)) as i64 >> (64 - size)) as u64
    }
}

impl CortexMCpu {
    /// Data processing: `111T 1110 opc1 opc2 | Vd 101 sz opc3 0 opc4`, T in
    /// bit 28 (Table A6-5).
    pub(super) fn fp_data_processing(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        if insn >> 28 == 0b1111 {
            return self.fp_v5_addition(insn);
        }
        // opc1 (bits 23:20) without D (bit 22), and the op bit of the
        // three-register forms (bit 6)
        let op = insn & (1 << 6) != 0;
        let operation = match ((insn >> 20) & 0b1011, op) {
            (0b0000, false) => Arithmetic::Vmla,
            (0b0000, true) => Arithmetic::Vmls,
            (0b0001, true) => Arithmetic::Vnmla,
            (0b0001, false) => Arithmetic::Vnmls,
            (0b0010, false) => Arithmetic::Vmul,
            (0b0010, true) => Arithmetic::Vnmul,
            (0b0011, false) => Arithmetic::Vadd,
            (0b0011, true) => Arithmetic::Vsub,
            (0b1000, false) => Arithmetic::Vdiv,
            (0b1001, true) => Arithmetic::Vfnma,
            (0b1001, false) => Arithmetic::Vfnms,
            (0b1010, false) => Arithmetic::Vfma,
            (0b1010, true) => Arithmetic::Vfms,
            (0b1011, _) => return self.fp_other(insn),
            _ => return Err(Fault::Undefined(insn)),
        };
        self.fp_arithmetic(insn, operation)
    }

    /// The three-register arithmetic (A7.7.225, 232 to 234, 238, 248, 250,
    /// 260), VMAXNM and VMINNM (A7.7.237).
    fn fp_arithmetic(&mut self, insn: u32, operation: Arithmetic) -> Result<CpuExit, Fault> {
        let registers = Registers::of(insn);
        let Registers { double, d, n, m } = registers;
        if !registers.exist(&[d, n, m]) {
            return Err(Fault::Undefined(insn));
        }
        let mut fpscr = self.fpscr();
        if double {
            let accumulator = f64::from_bits(self.d(d));
            let (a, b) = (f64::from_bits(self.d(n)), f64::from_bits(self.d(m)));
            let result = match operation {
                Arithmetic::Vmla => vfp::vmla_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vmls => vfp::vmls_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vnmla => vfp::vnmla_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vnmls => vfp::vnmls_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vnmul => vfp::vnmul_f64(a, b, &mut fpscr),
                Arithmetic::Vmul => vfp::vmul_f64(a, b, &mut fpscr),
                Arithmetic::Vadd => vfp::vadd_f64(a, b, &mut fpscr),
                Arithmetic::Vsub => vfp::vsub_f64(a, b, &mut fpscr),
                Arithmetic::Vdiv => vfp::vdiv_f64(a, b, &mut fpscr),
                Arithmetic::Vfma => vfp::vfma_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vfms => vfp::vfms_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vfnma => vfp::vfnma_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vfnms => vfp::vfnms_f64(accumulator, a, b, &mut fpscr),
                Arithmetic::Vmaxnm => vfp::vmaxnm_f64(a, b, &mut fpscr),
                Arithmetic::Vminnm => vfp::vminnm_f64(a, b, &mut fpscr),
            };
            self.set_d(d, result.to_bits());
        } else {
            let accumulator = f32::from_bits(self.s(d));
            let (a, b) = (f32::from_bits(self.s(n)), f32::from_bits(self.s(m)));
            let result = match operation {
                Arithmetic::Vmla => vfp::vmla_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vmls => vfp::vmls_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vnmla => vfp::vnmla_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vnmls => vfp::vnmls_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vnmul => vfp::vnmul_f32(a, b, &mut fpscr),
                Arithmetic::Vmul => vfp::vmul_f32(a, b, &mut fpscr),
                Arithmetic::Vadd => vfp::vadd_f32(a, b, &mut fpscr),
                Arithmetic::Vsub => vfp::vsub_f32(a, b, &mut fpscr),
                Arithmetic::Vdiv => vfp::vdiv_f32(a, b, &mut fpscr),
                Arithmetic::Vfma => vfp::vfma_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vfms => vfp::vfms_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vfnma => vfp::vfnma_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vfnms => vfp::vfnms_f32(accumulator, a, b, &mut fpscr),
                Arithmetic::Vmaxnm => vfp::vmaxnm_f32(a, b, &mut fpscr),
                Arithmetic::Vminnm => vfp::vminnm_f32(a, b, &mut fpscr),
            };
            self.set_s(d, result.to_bits());
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// `opc1 = 1x11`: VMOV (immediate) when opc3<0> is 0, else the
    /// two-register operations by opc2 (bits 19:16) and opc3<1> (bit 7).
    fn fp_other(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let opc2 = (insn >> 16) & 0xF;
        if insn & (1 << 6) == 0 {
            return self.fp_move_immediate(insn);
        }
        let op = insn & (1 << 7) != 0;
        match (opc2, op) {
            (0b0000, false) => self.fp_move_register(insn),
            (0b0000, true) => self.fp_unary(insn, Unary::Vabs),
            (0b0001, false) => self.fp_unary(insn, Unary::Vneg),
            (0b0001, true) => self.fp_unary(insn, Unary::Vsqrt),
            (0b0010 | 0b0011, _) => self.fp_convert_half(insn),
            (0b0100 | 0b0101, _) => self.fp_compare(insn),
            // VRINTR, VRINTZ (A7.7.255): the FPSCR's rounding, or towards zero
            (0b0110, false) => {
                let rounding = self.fpscr().rmode();
                self.fp_unary(insn, Unary::Vrint(rounding, false))
            }
            (0b0110, true) => self.fp_unary(insn, Unary::Vrint(RoundingMode::RoundZero, false)),
            // VRINTX (A7.7.254): the FPSCR's rounding, Inexact raised
            (0b0111, false) => {
                let rounding = self.fpscr().rmode();
                self.fp_unary(insn, Unary::Vrint(rounding, true))
            }
            (0b0111, true) => self.fp_convert_precision(insn),
            (0b1000, _) => self.fp_integer_to_float(insn),
            (0b1010 | 0b1011 | 0b1110 | 0b1111, _) => self.fp_convert_fixed(insn),
            (0b1100 | 0b1101, _) => self.fp_float_to_integer(insn),
            _ => Err(Fault::Undefined(insn)),
        }
    }

    /// VMOV (immediate) (A7.7.239): VFPExpandImm of imm4H:imm4L; bits 7 and
    /// 5 are zero.
    fn fp_move_immediate(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let registers = Registers::of(insn);
        let Registers { double, d, .. } = registers;
        if insn & 0xA0 != 0 || !registers.exist(&[d]) {
            return Err(Fault::Undefined(insn));
        }
        let imm8 = (((insn >> 16) & 0xF) << 4) | (insn & 0xF);
        if double {
            self.set_d(d, expand_imm64(imm8));
        } else {
            self.set_s(d, expand_imm32(imm8));
        }
        Ok(CpuExit::Continue)
    }

    /// VMOV (register) (A7.7.240).
    fn fp_move_register(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let registers = Registers::of(insn);
        let Registers { double, d, m, .. } = registers;
        if !registers.exist(&[d, m]) {
            return Err(Fault::Undefined(insn));
        }
        if double {
            self.set_d(d, self.d(m));
        } else {
            self.set_s(d, self.s(m));
        }
        Ok(CpuExit::Continue)
    }

    /// VABS, VNEG, VSQRT (A7.7.224, 249, 257) and the VRINTs (A7.7.253 to
    /// 255).
    fn fp_unary(&mut self, insn: u32, operation: Unary) -> Result<CpuExit, Fault> {
        let registers = Registers::of(insn);
        let Registers { double, d, m, .. } = registers;
        if !registers.exist(&[d, m]) {
            return Err(Fault::Undefined(insn));
        }
        let mut fpscr = self.fpscr();
        if double {
            let value = f64::from_bits(self.d(m));
            let result = match operation {
                Unary::Vabs => vfp::vabs_f64(value),
                Unary::Vneg => vfp::vneg_f64(value),
                Unary::Vsqrt => vfp::vsqrt_f64(value, &mut fpscr),
                Unary::Vrint(rounding, exact) => vfp::vrint_f64(value, rounding, exact, &mut fpscr),
            };
            self.set_d(d, result.to_bits());
        } else {
            let value = f32::from_bits(self.s(m));
            let result = match operation {
                Unary::Vabs => vfp::vabs_f32(value),
                Unary::Vneg => vfp::vneg_f32(value),
                Unary::Vsqrt => vfp::vsqrt_f32(value, &mut fpscr),
                Unary::Vrint(rounding, exact) => vfp::vrint_f32(value, rounding, exact, &mut fpscr),
            };
            self.set_s(d, result.to_bits());
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// VCMP, VCMPE (A7.7.226): FPSCR.NZCV from comparing the first operand
    /// with the second, or with +0.0 in encoding T2 (opc2 = 0101, bits 5 and
    /// 3:0 zero). E (bit 7) raises Invalid Operation for quiet NaNs too.
    fn fp_compare(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let registers = Registers::of(insn);
        let Registers { double, d, m, .. } = registers;
        let with_zero = insn & (1 << 16) != 0;
        let quiet_nan_exception = insn & (1 << 7) != 0;
        let operands_exist = if with_zero {
            registers.exist(&[d])
        } else {
            registers.exist(&[d, m])
        };
        if (with_zero && insn & 0x2F != 0) || !operands_exist {
            return Err(Fault::Undefined(insn));
        }
        let mut fpscr = self.fpscr();
        if double {
            let other = if with_zero {
                0.0
            } else {
                f64::from_bits(self.d(m))
            };
            let first = f64::from_bits(self.d(d));
            vfp::vcmp_f64_with_exception(first, other, quiet_nan_exception, &mut fpscr);
        } else {
            let other = if with_zero {
                0.0
            } else {
                f32::from_bits(self.s(m))
            };
            let first = f32::from_bits(self.s(d));
            vfp::vcmp_f32_with_exception(first, other, quiet_nan_exception, &mut fpscr);
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// VCVTB, VCVTT (A7.7.231): between half precision, in the bottom (T =
    /// 0) or top (T = 1, bit 7) halfword of a single register, and single or
    /// double precision; op (bit 16) set converts to half precision, leaving
    /// the other halfword.
    fn fp_convert_half(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let double = insn & (1 << 8) != 0;
        let to_half = insn & (1 << 16) != 0;
        let shift = if insn & (1 << 7) != 0 { 16 } else { 0 };
        let mut fpscr = self.fpscr();
        match (double, to_half) {
            (true, false) => {
                let (d, m) = (destination(insn, true), operand(insn, false));
                if d >= D_REGS {
                    return Err(Fault::Undefined(insn));
                }
                let half = (self.s(m) >> shift) as u16;
                let result = vfp::vcvt_f64_f16_bits_with_fpscr(half, &mut fpscr);
                self.set_d(d, result.to_bits());
            }
            (true, true) => {
                let (d, m) = (destination(insn, false), operand(insn, true));
                if m >= D_REGS {
                    return Err(Fault::Undefined(insn));
                }
                let half = vfp::vcvt_f16_bits_f64(f64::from_bits(self.d(m)), &mut fpscr);
                self.set_s(d, insert_half(self.s(d), half, shift));
            }
            (false, false) => {
                let (d, m) = (destination(insn, false), operand(insn, false));
                let half = (self.s(m) >> shift) as u16;
                let result = vfp::vcvt_f32_f16_bits_with_fpscr(half, &mut fpscr);
                self.set_s(d, result.to_bits());
            }
            (false, true) => {
                let (d, m) = (destination(insn, false), operand(insn, false));
                let half = vfp::vcvt_f16_bits_f32(f32::from_bits(self.s(m)), &mut fpscr);
                self.set_s(d, insert_half(self.s(d), half, shift));
            }
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// VCVT between double and single precision (A7.7.230): sz = 1 converts
    /// a double to a single.
    fn fp_convert_precision(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let double_to_single = insn & (1 << 8) != 0;
        let mut fpscr = self.fpscr();
        if double_to_single {
            let (d, m) = (destination(insn, false), operand(insn, true));
            if m >= D_REGS {
                return Err(Fault::Undefined(insn));
            }
            let result = vfp::vcvt_f32_f64(f64::from_bits(self.d(m)), &mut fpscr);
            self.set_s(d, result.to_bits());
        } else {
            let (d, m) = (destination(insn, true), operand(insn, false));
            if d >= D_REGS {
                return Err(Fault::Undefined(insn));
            }
            let result = vfp::vcvt_f64_f32_with_fpscr(f32::from_bits(self.s(m)), &mut fpscr);
            self.set_d(d, result.to_bits());
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// VCVT from a 32-bit integer in a single register (A7.7.228, opc2 =
    /// 000): signed when op (bit 7) is set, rounded by the FPSCR.
    fn fp_integer_to_float(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let double = insn & (1 << 8) != 0;
        let signed = insn & (1 << 7) != 0;
        let d = destination(insn, double);
        let integer = self.s(operand(insn, false));
        if double && d >= D_REGS {
            return Err(Fault::Undefined(insn));
        }
        let mut fpscr = self.fpscr();
        if double {
            let result = if signed {
                vfp::vcvt_f64_s32_with_fpscr(integer as i32, &mut fpscr)
            } else {
                vfp::vcvt_f64_u32_with_fpscr(integer, &mut fpscr)
            };
            self.set_d(d, result.to_bits());
        } else {
            let result = if signed {
                vfp::vcvt_f32_s32_with_fpscr(integer as i32, &mut fpscr)
            } else {
                vfp::vcvt_f32_u32_with_fpscr(integer, &mut fpscr)
            };
            self.set_s(d, result.to_bits());
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// VCVT, VCVTR to a 32-bit integer in a single register (A7.7.228,
    /// opc2 = 10x): signed when opc2<0> (bit 16) is set; op (bit 7) set
    /// rounds towards zero (VCVT), clear by the FPSCR (VCVTR).
    fn fp_float_to_integer(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let mut fpscr = self.fpscr();
        let rounding = if insn & (1 << 7) != 0 {
            RoundingMode::RoundZero
        } else {
            fpscr.rmode()
        };
        self.fp_to_integer(insn, rounding, insn & (1 << 16) != 0, &mut fpscr)?;
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// Converts the operand (`M:Vm` or `Vm:M`) to a 32-bit integer in `Vd:D`.
    fn fp_to_integer(
        &mut self,
        insn: u32,
        rounding: RoundingMode,
        signed: bool,
        fpscr: &mut Fpscr,
    ) -> Result<(), Fault> {
        let double = insn & (1 << 8) != 0;
        let (d, m) = (destination(insn, false), operand(insn, double));
        let result = if double {
            if m >= D_REGS {
                return Err(Fault::Undefined(insn));
            }
            let value = f64::from_bits(self.d(m));
            if signed {
                vfp::vcvt_s32_f64_round(value, rounding, fpscr) as u32
            } else {
                vfp::vcvt_u32_f64_round(value, rounding, fpscr)
            }
        } else {
            let value = f32::from_bits(self.s(m));
            if signed {
                vfp::vcvt_s32_f32_round(value, rounding, fpscr) as u32
            } else {
                vfp::vcvt_u32_f32_round(value, rounding, fpscr)
            }
        };
        self.set_s(d, result);
        Ok(())
    }

    /// VCVT between floating point and fixed point in place (A7.7.229):
    /// `1110 1110 1D11 1o1U | Vd 101 sf sx 1 i 0 imm4`. op (bit 18) converts
    /// to fixed point, rounding towards zero, the result zero- or
    /// sign-extended; from fixed point rounds to nearest. The fixed-point
    /// value has `size` bits (16, or 32 with sx) and `size - imm4:i`
    /// fraction bits; fewer than 0 is UNPREDICTABLE, UNDEFINED here.
    fn fp_convert_fixed(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let double = insn & (1 << 8) != 0;
        let to_fixed = insn & (1 << 18) != 0;
        let unsigned = insn & (1 << 16) != 0;
        let size: u32 = if insn & (1 << 7) != 0 { 32 } else { 16 };
        let imm5 = ((insn & 0xF) << 1) | ((insn >> 5) & 1);
        let d = destination(insn, double);
        let Some(fraction_bits) = size.checked_sub(imm5) else {
            return Err(Fault::Undefined(insn));
        };
        if double && d >= D_REGS {
            return Err(Fault::Undefined(insn));
        }
        let mut fpscr = self.fpscr();
        let (raw, fp_bits) = if double {
            (self.d(d), 64)
        } else {
            (u64::from(self.s(d)), 32)
        };
        let (result, status) = SmirInterpreter::arm_eval_a32_fixed_conversion(
            raw,
            fp_bits,
            size,
            fraction_bits,
            !unsigned,
            to_fixed,
            fpscr.bits(),
        );
        fpscr = Fpscr::from_bits(fpscr.bits() | status);
        let result = if to_fixed {
            extend_fixed(result, size, unsigned)
        } else {
            result
        };
        if double {
            self.set_d(d, result);
        } else {
            self.set_s(d, result as u32);
        }
        self.set_fpscr(fpscr);
        Ok(CpuExit::Continue)
    }

    /// The FPv5 additions, 0b1111 in bits 31:28 (Table A6-5, T = 1).
    fn fp_v5_addition(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        if self.in_it_block() {
            return Err(Fault::Undefined(insn));
        }
        let op = insn & (1 << 6) != 0;
        if insn & (1 << 23) == 0 {
            return if op {
                Err(Fault::Undefined(insn))
            } else {
                self.fp_select(insn)
            };
        }
        match ((insn >> 20) & 3, (insn >> 18) & 3) {
            // VMAXNM, VMINNM (A7.7.237): op selects the minimum
            (0b00, _) if op => self.fp_arithmetic(insn, Arithmetic::Vminnm),
            (0b00, _) => self.fp_arithmetic(insn, Arithmetic::Vmaxnm),
            // VRINTA, VRINTN, VRINTP, VRINTM (A7.7.253): bits 7:6 = 01
            (0b11, 0b10) if (insn >> 6) & 3 == 0b01 => {
                let rounding = directed_rounding(insn);
                self.fp_unary(insn, Unary::Vrint(rounding, false))
            }
            // VCVTA, VCVTN, VCVTP, VCVTM (A7.7.227): op set, signed when
            // bit 7 is set
            (0b11, 0b11) if op => {
                let mut fpscr = self.fpscr();
                let rounding = directed_rounding(insn);
                self.fp_to_integer(insn, rounding, insn & (1 << 7) != 0, &mut fpscr)?;
                self.set_fpscr(fpscr);
                Ok(CpuExit::Continue)
            }
            _ => Err(Fault::Undefined(insn)),
        }
    }

    /// VSEL (A7.7.256): the first operand if the condition cc:(cc<1> XOR
    /// cc<0>):0 holds in APSR (EQ, VS, GE, GT for cc = 0 to 3), else the
    /// second.
    fn fp_select(&mut self, insn: u32) -> Result<CpuExit, Fault> {
        let registers = Registers::of(insn);
        let Registers { double, d, n, m } = registers;
        if !registers.exist(&[d, n, m]) {
            return Err(Fault::Undefined(insn));
        }
        let cc = (insn >> 20) & 3;
        let condition = (cc << 2) | ((((cc >> 1) ^ cc) & 1) << 1);
        let source = if self.condition_passed(condition) {
            n
        } else {
            m
        };
        if double {
            self.set_d(d, self.d(source));
        } else {
            self.set_s(d, self.s(source));
        }
        Ok(CpuExit::Continue)
    }
}

/// `VFPExpandImm(imm8, 32)`: sign imm8<7>, exponent NOT(imm8<6>) then
/// imm8<6> five times then imm8<5:4>, fraction imm8<3:0> then 19 zeros.
fn expand_imm32(imm8: u32) -> u32 {
    let b6 = (imm8 >> 6) & 1;
    ((imm8 >> 7) << 31)
        | ((b6 ^ 1) << 30)
        | ((b6 * 0b1_1111) << 25)
        | (((imm8 >> 4) & 3) << 23)
        | ((imm8 & 0xF) << 19)
}

/// `VFPExpandImm(imm8, 64)`: sign imm8<7>, exponent NOT(imm8<6>) then
/// imm8<6> eight times then imm8<5:4>, fraction imm8<3:0> then 48 zeros.
fn expand_imm64(imm8: u32) -> u64 {
    let imm8 = u64::from(imm8);
    let b6 = (imm8 >> 6) & 1;
    ((imm8 >> 7) << 63)
        | ((b6 ^ 1) << 62)
        | ((b6 * 0xFF) << 54)
        | (((imm8 >> 4) & 3) << 52)
        | ((imm8 & 0xF) << 48)
}

#[cfg(test)]
#[path = "fp_data_tests.rs"]
mod tests;
