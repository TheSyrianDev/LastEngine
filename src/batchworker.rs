//! Server-side batched worker (phase 2 step 4). One thread owns the GPU step; requests arrive on a channel,
//! are admitted into free rows between steps, and every generated token and finish is sent back per request.
//! Pure logic here: the GPU backend passes a `step` closure (decode_batch + greedy_rows). See batchplan.rs.
use crate::batchplan::{step_once, Finish, Seq, StepInput};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

/// A request admitted to the worker: its id, first token (from prefill) and generation limit.
pub struct Job { pub id: u64, pub seq: Seq }

/// Events sent back to the request's owner.
#[derive(Debug, PartialEq, Eq)]
pub enum Event { Token(u64, u32), Done(u64, Finish) }

/// Run until the job channel is closed and every admitted sequence has finished.
/// Admission: while rows are free, take jobs without blocking; block only when idle.
pub fn run<F>(jobs: &Receiver<Job>, out: &Sender<Event>, eos: &[u32], ctx_len: usize, max_rows: usize, mut step: F)
where F: FnMut(&StepInput) -> Vec<u32> {
    let mut seqs: Vec<(u64, Seq)> = vec![];
    let mut open = true;
    loop {
        // Admit.
        while seqs.len() < max_rows && open {
            let got = if seqs.is_empty() { jobs.recv().map_err(|_| TryRecvError::Disconnected) } else { jobs.try_recv() };
            match got {
                Ok(j) => seqs.push((j.id, j.seq)),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { open = false; break; }
            }
        }
        if seqs.is_empty() { if !open { return; } else { continue; } }
        let ids: Vec<u64> = seqs.iter().map(|(i, _)| *i).collect();
        let mut emitted: Vec<(u64, u32)> = vec![];
        let fin = {
            let mut wrapped = |plan: &StepInput| {
                let toks = step(plan);
                for (k, &row) in plan.rows.iter().enumerate() { emitted.push((ids[row], toks[k])); }
                toks
            };
            step_once(&mut seqs, eos, ctx_len, max_rows, &mut wrapped)
        };
        for (id, t) in emitted { let _ = out.send(Event::Token(id, t)); }
        if let Some(fin) = fin { for (id, r) in fin { let _ = out.send(Event::Done(id, r)); } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    fn seq(slot: usize, pos: usize, tok: u32, max: usize) -> Seq { Seq { slot, pos, next_tok: tok, generated: 0, max_tokens: max } }

    #[test]
    fn worker_streams_tokens_and_finishes_each_request() {
        let (jtx, jrx) = channel();
        let (otx, orx) = channel();
        jtx.send(Job { id: 1, seq: seq(0, 10, 5, 2) }).unwrap();
        jtx.send(Job { id: 2, seq: seq(1, 20, 9, 1) }).unwrap();
        drop(jtx);
        run(&jrx, &otx, &[], 4096, 4, |p| p.toks.iter().map(|t| t + 1).collect());
        drop(otx);
        let ev: Vec<Event> = orx.iter().collect();
        assert_eq!(ev.iter().filter(|e| matches!(e, Event::Token(1, _))).count(), 2);
        assert_eq!(ev.iter().filter(|e| matches!(e, Event::Token(2, _))).count(), 1);
        assert!(ev.contains(&Event::Done(2, Finish::Length)));
        assert!(ev.contains(&Event::Done(1, Finish::Length)));
        // Token values: request 1 starts at 5 -> 6, 7; request 2 at 9 -> 10.
        assert!(ev.contains(&Event::Token(1, 6)) && ev.contains(&Event::Token(1, 7)) && ev.contains(&Event::Token(2, 10)));
    }

    #[test]
    fn worker_exits_when_channel_closed_and_idle() {
        let (jtx, jrx) = channel::<Job>();
        let (otx, _orx) = channel();
        drop(jtx);
        run(&jrx, &otx, &[], 4096, 4, |_| panic!("no step when idle"));
    }
}
