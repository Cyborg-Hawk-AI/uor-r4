//! WebAssembly serving surface for the D11 stack runtime (#1172).
//!
//! This module is the load/session/step/save/restore export set the Studio
//! worker drives. It exists only with `serving-wasm` on `wasm32`, and it owns
//! no model logic: every call forwards to the preserved engine
//! ([`crate::stack::IntegerStackModel::parse`], [`IntegerStackSession::step`],
//! [`stack_argmax`], [`IntegerStackSession::save_state`]). Greedy selection
//! is the engine's own integer argmax (first on ties), so save → terminate →
//! restore reproduces a greedy stream byte-for-byte as long as the artifact
//! bytes are the same sealed ones.
//!
//! **Session pairing without `unsafe`.** This crate is
//! `#![forbid(unsafe_code)]`, so a session (which borrows its model) cannot
//! be paired with its model in one owning struct. The serving surface
//! therefore keeps one slot per loaded model: the model is `Box::leak`'d to
//! `'static` (one allocation per loaded model, counted in
//! [`wasm_leaked_models`]) and its session lives beside it in a
//! `thread_local!` table. A module generation loads models; it does not
//! unload them (the Studio worker is itself the lifetime boundary —
//! terminate the worker, free the world).
//!
//! R2 discipline (their audit `scripts/audit_zero_matmul_wasm.py`): nothing
//! here multiplies except through the engine's own table paths. The margin
//! arithmetic is subtraction of adjacent maxima. The slot table avoids
//! `Mutex` (its lock path carries an `i32.mul`) in favor of `thread_local!`
//! interior mutability, valid in the single-threaded module. The counters
//! are plain `AtomicU64`s; the global allocator is not replaced.

#![cfg(all(target_arch = "wasm32", feature = "serving-wasm"))]

use crate::stack::{stack_argmax, IntegerStackModel, StackError};
use core::sync::atomic::{AtomicU64, Ordering};
use std::cell::RefCell;

/// One loaded model (leaked, `'static`) plus its live session.
struct Slot {
    model: &'static IntegerStackModel,
    session: crate::stack::IntegerStackSession<'static>,
}

/// Handle-space capacity. 8 simultaneous loaded models is past any Studio
/// use (one artifact per worker today).
const MAX_SLOTS: usize = 8;

thread_local! {
    static SLOTS: RefCell<Vec<Option<Slot>>> = const { RefCell::new(Vec::new()) };
}

static LEAKED_MODELS: AtomicU64 = AtomicU64::new(0);
static ALLOC_EVENTS: AtomicU64 = AtomicU64::new(0);

fn take_error(e: StackError) -> String {
    format!("{e:?}")
}

/// Load a `UORLUT01` stack artifact from bytes (the host fetches it).
/// Returns the model/session handle, or an error string.
pub fn wasm_load_stack(artifact: &[u8]) -> Result<u32, String> {
    let model = IntegerStackModel::parse(artifact).map_err(take_error)?;
    let model_ref: &'static IntegerStackModel = Box::leak(Box::new(model));
    LEAKED_MODELS.fetch_add(1, Ordering::Relaxed);
    let session = model_ref.session();
    ALLOC_EVENTS.fetch_add(1, Ordering::Relaxed);
    SLOTS.with(|t| {
        let mut table = t.borrow_mut();
        if table.len() < MAX_SLOTS && table.iter().all(|s| s.is_some()) {
            table.push(None);
        }
        for (i, slot) in table.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(Slot {
                    model: model_ref,
                    session,
                });
                return Ok(i as u32);
            }
        }
        Err("no free session slots".to_string())
    })
}

/// The result of one step.
pub struct StepOut {
    /// The greedy next token (integer argmax, first on ties).
    pub next: u32,
    /// top1 − top2 of the logits, Q16 units, adjacent-maxima subtraction only.
    pub margin_q16: i32,
    /// Context position after the step.
    pub position: u32,
}

/// Feed one token at the session's next position.
pub fn wasm_step(handle: u32, token: u32) -> Result<StepOut, String> {
    SLOTS.with(|t| {
        let mut table = t.borrow_mut();
        let slot = table
            .get_mut(handle as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| format!("unknown session handle {handle}"))?;
        let logits = slot.session.step(token).map_err(take_error)?;
        let next = stack_argmax(logits) as u32;
        let (mut top1, mut top2) = (i32::MIN, i32::MIN);
        for &v in logits.iter() {
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        Ok(StepOut {
            next,
            margin_q16: top1.wrapping_sub(top2),
            position: slot.session.position() as u32,
        })
    })
}

/// Serialize the session state (position + recurrence + read caches).
pub fn wasm_save(handle: u32) -> Result<Vec<u8>, String> {
    SLOTS.with(|t| {
        let mut table = t.borrow_mut();
        let slot = table
            .get_mut(handle as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| format!("unknown session handle {handle}"))?;
        let state = slot.session.save_state();
        serde_json::to_vec(&state).map_err(|e| e.to_string())
    })
}

/// Restore a serialized session into the same loaded model (the
/// reload-identical test: save → fresh worker → restore → byte-identical
/// greedy stream).
pub fn wasm_restore(handle: u32, state: &[u8]) -> Result<(), String> {
    SLOTS.with(|t| {
        let mut table = t.borrow_mut();
        let slot = table
            .get_mut(handle as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| format!("unknown session handle {handle}"))?;
        let parsed =
            serde_json::from_slice(state).map_err(|e: serde_json::Error| e.to_string())?;
        slot.session.restore_state(&parsed).map_err(take_error)
    })
}

/// Weights touched per token (their R3 per-token parameter-reads number).
pub fn wasm_weights_per_token(handle: u32) -> Result<u64, String> {
    SLOTS.with(|t| {
        let table = t.borrow();
        let slot = table
            .get(handle as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("unknown session handle {handle}"))?;
        Ok(slot.model.weights_per_token())
    })
}

/// Allocation events since module init (loads only; the step loop performs
/// none — every buffer a step uses was allocated at `session()` creation).
/// Two consecutive `step` calls with an unchanged counter prove it.
pub fn wasm_alloc_events() -> u64 {
    ALLOC_EVENTS.load(Ordering::Relaxed)
}

/// Models leaked for the module lifetime (one per `wasm_load_stack`).
pub fn wasm_leaked_models() -> u64 {
    LEAKED_MODELS.load(Ordering::Relaxed)
}
