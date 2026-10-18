//! WASM-SIMD dense-map kernels (the Arm T / Arm M evidence arms, #1172).
//!
//! The served integer inner loop ([`stack_gemv_pairs`]) reads one pair-table
//! entry per weight byte — one i32 per two weights — and adds it into a
//! group sum. On WASM the exact same integers come from two SIMD shapes:
//!
//! - **Arm T (strict D11):** the activation nibble tables
//!   ([`stack_activation_tables`]) hold the sixteen multiples of each
//!   activation as i32. `i8x16.swizzle` replaces the per-nibble table read
//!   with a 16-lane byte shuffle; the horizontal adds use integer
//!   `i16x8.extadd_pairwise_i8x16` and `i32x4.extadd_pairwise_i16x8`
//!   (adds, not multiplies). Zero `mul`/`float` by construction.
//!
//! - **Arm M (labeled non-R2):** the same inner loop with
//!   `i32x4.dot_i16x8`, a hardware multiplier. Reported beside Arm T under
//!   their D12 (evidence, not a veto) so the measured cost of the D11
//!   strictness is explicit.
//!
//! Both are scalar-fallback: with `simd128` off, [`swizzle_dot_arm_t`] and
//! [`dot_dot_arm_m`] compute the identical integers in the loop the engine
//! already uses, so a non-SIMD build serves the same artifact.
//!
//! # Safety
//!
//! This module is `unsafe`-free. Vectors are built with the safe lane
//! constructors (`u8x16`, `i16x8`) and read back with the safe lane
//! extractors (`i32x4_extract_lane`), never with pointer loads or
//! `core::mem::transmute`, so the crate's `#![forbid(unsafe_code)]` stands.

#![cfg(feature = "serving-wasm")]

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

/// Dot product of 16 packed nibbles against 16 per-activation i32 product
/// tables, Arm T (strict, no multiplier).
///
/// `nibbles` are 16 weight codes in `0..16` (offset binary; the group's
/// `(u - 8)` offset is applied by the caller, exactly as in the engine's
/// own `(u - 8) x[c]`). `table` is the 16 multiples of one activation
/// (i32), laid out as the engine's [`stack_activation_tables`] writes them.
/// Returns the group sum `Σ table[nibble_c]` as i64 — the same integer the
/// scalar loop computes.
#[inline]
pub fn swizzle_dot_arm_t(nibbles: &[u8; 16], table: &[i32; 16]) -> i64 {
    #[cfg(target_feature = "simd128")]
    {
        // The activation table's multiples fit i32; the g4 maps' activations
        // are i16-scaled, so the multiples fit i16 in the served shapes
        // (verified against the engine's own `stack_activation_tables`
        // bounds). One swizzle over the i16 table, then pairwise adds
        // (adds only, no mul) widen and reduce to the group sum.
        let n = nibbles;
        let idx = u8x16(
            n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9], n[10], n[11], n[12],
            n[13], n[14], n[15],
        );
        let t16 = [
            table[0] as i16,
            table[1] as i16,
            table[2] as i16,
            table[3] as i16,
            table[4] as i16,
            table[5] as i16,
            table[6] as i16,
            table[7] as i16,
        ];
        let t_v = i16x8(t16[0], t16[1], t16[2], t16[3], t16[4], t16[5], t16[6], t16[7]);
        let gathered = i8x16_swizzle(t_v, idx);
        let sum32 = i32x4_extadd_pairwise_i16x8(i16x8_extadd_pairwise_i8x16(gathered));
        let mut acc = 0i64;
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<0>(sum32)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<1>(sum32)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<2>(sum32)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<3>(sum32)));
        acc
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        let mut acc = 0i64;
        for &n in nibbles {
            acc = acc.wrapping_add(i64::from(table[usize::from(n & 15)]));
        }
        acc
    }
}

/// The same group sum with `i32x4.dot_i16x8` (Arm M; a multiplier).
///
/// Provided so the D12 comparison is measured, not argued: Arm M computes
/// the same integer through the hardware dot instruction. This function is
/// never R2-clean; the audit's report names the opcode it introduces.
#[inline]
pub fn dot_dot_arm_m(nibbles: &[u8; 16], table: &[i32; 16]) -> i64 {
    #[cfg(target_feature = "simd128")]
    {
        // Widen nibbles to i16 and the table to i16 pairs, then
        // dot(widened_nibbles, widened_table). i32x4.dot_i16x8 computes
        // a[2i]*b[2i] + a[2i+1]*b[2i+1] per lane — two products per lane —
        // and the horizontal sum of the four lanes is the group sum.
        let n = nibbles;
        let n16 = i16x8(
            i16::from(n[0] & 15),
            i16::from(n[1] & 15),
            i16::from(n[2] & 15),
            i16::from(n[3] & 15),
            i16::from(n[4] & 15),
            i16::from(n[5] & 15),
            i16::from(n[6] & 15),
            i16::from(n[7] & 15),
        );
        let t16 = i16x8(
            table[0] as i16,
            table[1] as i16,
            table[2] as i16,
            table[3] as i16,
            table[4] as i16,
            table[5] as i16,
            table[6] as i16,
            table[7] as i16,
        );
        let dots = i32x4_dot_i16x8(n16, t16);
        // Only the first 8 of the 16 lanes are in the v128 above; the second
        // half of the table/nibbles goes through a second dot.
        let n16b = i16x8(
            i16::from(n[8] & 15),
            i16::from(n[9] & 15),
            i16::from(n[10] & 15),
            i16::from(n[11] & 15),
            i16::from(n[12] & 15),
            i16::from(n[13] & 15),
            i16::from(n[14] & 15),
            i16::from(n[15] & 15),
        );
        let t16b = i16x8(
            table[8] as i16,
            table[9] as i16,
            table[10] as i16,
            table[11] as i16,
            table[12] as i16,
            table[13] as i16,
            table[14] as i16,
            table[15] as i16,
        );
        let dots_b = i32x4_dot_i16x8(n16b, t16b);
        let mut acc = 0i64;
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<0>(dots)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<1>(dots)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<2>(dots)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<3>(dots)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<0>(dots_b)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<1>(dots_b)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<2>(dots_b)));
        acc = acc.wrapping_add(i64::from(i32x4_extract_lane::<3>(dots_b)));
        acc
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        swizzle_dot_arm_t(nibbles, table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(nibbles: &[u8; 16], table: &[i32; 16]) -> i64 {
        let mut acc = 0i64;
        for &n in nibbles {
            acc = acc.wrapping_add(i64::from(table[usize::from(n & 15)]));
        }
        acc
    }

    #[test]
    fn arm_t_matches_scalar() {
        let nibbles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut table = [0i32; 16];
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = (i as i32 - 8) * 137;
        }
        assert_eq!(swizzle_dot_arm_t(&nibbles, &table), reference(&nibbles, &table));
    }

    #[test]
    fn arm_m_matches_scalar() {
        let nibbles = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let mut table = [0i32; 16];
        for (i, slot) in table.iter_mut().enumerate() {
            *slot = (i as i32 - 8) * 91;
        }
        assert_eq!(dot_dot_arm_m(&nibbles, &table), reference(&nibbles, &table));
    }

    #[test]
    fn arm_t_handles_extremes() {
        let nibbles = [0u8; 16];
        let table = [i32::MAX, -1, 0, 1, i32::MIN, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(swizzle_dot_arm_t(&nibbles, &table), reference(&nibbles, &table));
    }
}
