//! Byte-level BPE tokenizer (GPT-2 style) built from GGUF metadata, with the Qwen pre-tokenizer.
use crate::gguf::{Gguf, Value};
use fancy_regex::Regex;
use std::collections::HashMap;

pub struct Tokenizer {
    pub tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    specials: Vec<(String, u32)>,
    re: Regex,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
}

const LLAMA3_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN_PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut enc = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        enc[b as usize] = if keep { char::from_u32(b).unwrap() } else { n += 1; char::from_u32(255 + n).unwrap() };
    }
    let dec = enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
    (enc, dec)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let strs = |k: &str| -> Result<Vec<String>, String> {
            match g.kv.get(k) {
                Some(Value::Arr(a)) => Ok(a.iter().filter_map(|v| v.as_str().map(String::from)).collect()),
                _ => Err(format!("missing {k}")),
            }
        };
        let tokens = strs("tokenizer.ggml.tokens")?;
        let merges = strs("tokenizer.ggml.merges")?;
        let types: Vec<u64> = match g.kv.get("tokenizer.ggml.token_type") {
            Some(Value::Arr(a)) => a.iter().map(|v| match v { Value::I32(x) => *x as u64, v => v.as_u64().unwrap_or(1) }).collect(),
            _ => vec![1; tokens.len()],
        };
        let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let mut ranks = HashMap::new();
        for (i, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') { ranks.insert((a.to_string(), b.to_string()), i as u32); }
        }
        let mut specials: Vec<(String, u32)> = tokens.iter().enumerate()
            .filter(|(i, _)| matches!(types.get(*i), Some(3) | Some(4)))
            .map(|(i, t)| (t.clone(), i as u32)).collect();
        specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
        let (byte_enc, byte_dec) = byte_tables();
        let id = |k: &str| g.kv.get(k).and_then(Value::as_u64).map(|v| v as u32);
        Ok(Tokenizer {
            tokens, ids, ranks, specials, re: Regex::new(match g.kv.get("tokenizer.ggml.pre").and_then(Value::as_str) { Some("lfm2") | Some("minicpm5") | Some("llama3") | Some("llama-bpe") => LLAMA3_PRE, _ => QWEN_PRE }).map_err(|e| e.to_string())?,
            byte_enc, byte_dec,
            bos: id("tokenizer.ggml.bos_token_id"), eos: id("tokenizer.ggml.eos_token_id"),
        })
    }

    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.bytes().map(|b| self.byte_enc[b as usize].to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) { best = Some((r, i)); }
                }
            }
            let Some((_, i)) = best else { break };
            let nxt = parts.remove(i + 1);
            parts[i].push_str(&nxt);
        }
        for p in parts { if let Some(&id) = self.ids.get(&p) { out.push(id); } }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.re.find_iter(text).flatten() { self.bpe(m.as_str(), out); }
    }

    /// Encode text; special tokens appearing literally in the text map to their ids.
    pub fn special(&self, s: &str) -> Option<u32> { self.specials.iter().find(|(t, _)| t == s).map(|(_, i)| *i) }
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            let mut first: Option<(usize, &str, u32)> = None;
            for (t, id) in &self.specials {
                if let Some(p) = rest.find(t.as_str()) {
                    if first.map_or(true, |(fp, ft, _)| p < fp || (p == fp && t.len() > ft.len())) { first = Some((p, t, *id)); }
                }
            }
            match first {
                Some((p, t, id)) => { self.encode_plain(&rest[..p], &mut out); out.push(id); rest = &rest[p + t.len()..]; }
                None => { self.encode_plain(rest, &mut out); break 'outer; }
            }
        }
        out
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &i in ids {
            let Some(t) = self.tokens.get(i as usize) else { continue };
            if self.specials.iter().any(|(s, _)| s == t) { out.extend_from_slice(t.as_bytes()); continue; }
            for c in t.chars() { if let Some(&b) = self.byte_dec.get(&c) { out.push(b); } }
        }
        out
    }
}
