//! Request scheduler for LE_BATCH=1 (S1). One thread owns the generator; requests arrive on a channel and
//! tokens stream back through each request's own Sender. The generator is injected so tests run without a GPU.
//! Note: src/sched.rs is the earlier FIFO slot scheduler; this module is separate on purpose.
use std::sync::mpsc::{Receiver, Sender};

#[derive(Clone, Debug, PartialEq)]
pub struct Sampling { pub temperature: f32, pub top_p: f32 }

#[derive(Debug, PartialEq)]
pub enum Token { Tok(u32), Done { n: usize } }

pub struct Request { pub prompt_ids: Vec<u32>, pub max_tokens: usize, pub sampling: Sampling, pub tx: Sender<Token> }

/// Pulls requests until the channel closes. `gen` runs one request, calling `on_tok` per generated token.
/// Sends Token::Done{n} when finished. A dropped receiver (client gone) stops that request's stream only.
pub fn run<G>(rx: Receiver<Request>, mut gen: G)
where G: FnMut(&Request, &mut dyn FnMut(u32) -> bool) -> usize {
    while let Ok(req) = rx.recv() {
        let tx = req.tx.clone();
        let mut alive = true;
        let n = gen(&req, &mut |t| { if tx.send(Token::Tok(t)).is_err() { alive = false; } alive });
        let _ = tx.send(Token::Done { n });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    #[test]
    fn streams_tokens_then_done() {
        let (qtx, qrx) = channel();
        let (ttx, trx) = channel();
        qtx.send(Request { prompt_ids: vec![1, 2], max_tokens: 3, sampling: Sampling { temperature: 0.0, top_p: 1.0 }, tx: ttx }).unwrap();
        drop(qtx);
        run(qrx, |req, on| { let mut n = 0; for t in 0..req.max_tokens as u32 { if !on(t + 10) { break; } n += 1; } n });
        let got: Vec<Token> = trx.iter().collect();
        assert_eq!(got, vec![Token::Tok(10), Token::Tok(11), Token::Tok(12), Token::Done { n: 3 }]);
    }
    #[test]
    fn two_requests_in_order() {
        let (qtx, qrx) = channel();
        let mut outs = vec![];
        for k in 0..2u32 {
            let (ttx, trx) = channel();
            qtx.send(Request { prompt_ids: vec![k], max_tokens: 1, sampling: Sampling { temperature: 0.0, top_p: 1.0 }, tx: ttx }).unwrap();
            outs.push(trx);
        }
        drop(qtx);
        run(qrx, |req, on| { on(req.prompt_ids[0] * 100); 1 });
        assert_eq!(outs[0].iter().collect::<Vec<_>>(), vec![Token::Tok(0), Token::Done { n: 1 }]);
        assert_eq!(outs[1].iter().collect::<Vec<_>>(), vec![Token::Tok(100), Token::Done { n: 1 }]);
    }
}
