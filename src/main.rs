//! lastengine: `lastengine serve --model X.gguf --addr 127.0.0.1:PORT [flags]`
mod backend;
mod cli;
mod cpu_ref;
mod gguf;
mod gpu;
mod local;
mod server;
mod quant;
mod template;
mod tokenizer;

use clap::Parser;

#[tokio::main]
async fn main() {
    let cli::Cli { cmd } = cli::Cli::parse();
    match cmd {
        cli::Cmd::Inspect { path } => match gguf::Gguf::open(&path) {
            Ok(g) => {
                println!("GGUF v{} arch={:?} tensors={} data_start={}", g.version, g.arch(), g.tensors.len(), g.data_start);
                let mut by: std::collections::BTreeMap<(u32, String), (usize, u64)> = Default::default();
                for t in &g.tensors {
                    let base = t.name.split('.').filter(|p| p.parse::<u32>().is_err()).collect::<Vec<_>>().join(".");
                    let e = by.entry((t.ggml_type, base)).or_default();
                    e.0 += 1; e.1 += t.dims.iter().product::<u64>();
                }
                for ((ty, name), (n, el)) in &by { println!("  type {ty:>2} x{n:<3} {name} ({el} elems)"); }
                for (k, v) in &g.kv {
                    if !matches!(v, gguf::Value::Arr(_)) { println!("  {k} = {v:?}"); }
                }
            }
            Err(e) => { eprintln!("lastengine: {path}: {e}"); std::process::exit(1); }
        },
        cli::Cmd::Tokenize { path, text } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let t = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            let ids = t.encode(&text);
            println!("{ids:?}");
            println!("{:?}", String::from_utf8_lossy(&t.decode_bytes(&ids)));
        }
        cli::Cmd::Template { path } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let t = g.kv.get("tokenizer.chat_template").and_then(|v| v.as_str()).expect("no template");
            let msgs: Vec<serde_json::Value> = serde_json::from_str(r#"[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Bye"}]"#).unwrap();
            println!("{}", template::render(t, &msgs, None, true, "", "").unwrap_or_else(|e| e));
        }
        cli::Cmd::Dequant { path, name, n, out } => {
            use std::io::{Read, Seek, Write};
            let g = gguf::Gguf::open(&path).expect("open");
            let t = g.tensors.iter().find(|t| t.name == name).expect("tensor");
            let (vb, bb) = quant::block_info(t.ggml_type).unwrap();
            let nb = n / vb;
            let mut f = std::fs::File::open(&path).unwrap();
            f.seek(std::io::SeekFrom::Start(g.data_start + t.offset)).unwrap();
            let mut buf = vec![0u8; nb * bb];
            f.read_exact(&mut buf).unwrap();
            let mut v = vec![0f32; nb * vb];
            quant::dequant_row(t.ggml_type, &buf, &mut v);
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::File::create(out).unwrap().write_all(&bytes).unwrap();
        }
        cli::Cmd::Ref { path, prompt, top } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let tk = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            let m = cpu_ref::Model::load(&path, &g);
            let ids = tk.encode(&prompt);
            let mut st = m.new_state();
            let mut logits = vec![];
            for (i, id) in ids.iter().enumerate() { logits = m.forward(&mut st, *id, i); }
            let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
            let mut ix: Vec<usize> = (0..logits.len()).collect();
            ix.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
            for i in ix.iter().take(top) { println!("{} {:?} logprob {:.4}", i, tk.decode_bytes(&[*i as u32]).iter().map(|b| *b as char).collect::<String>(), logits[*i] - lse); }
        }
        cli::Cmd::Gpu { path, prompt, top, margin } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let tk = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            let m = std::sync::Arc::new(cpu_ref::Model::load(&path, &g));
            let mut gp = gpu::Gpu::new(m, std::env::var("LE_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048), margin);
            let ids = tk.encode(&prompt);
            let t0 = std::time::Instant::now();
            let mut logits = vec![];
            logits = gp.prefill(&ids);
            println!("prefill {} tok in {:.2}s", ids.len(), t0.elapsed().as_secs_f32());
            let mut ix: Vec<usize> = (0..logits.len()).collect();
            let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
            ix.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
            for i in ix.iter().take(top) { println!("{} {:?} logprob {:.4}", i, tk.decode_bytes(&[*i as u32]).iter().map(|b| *b as char).collect::<String>(), logits[*i] - lse); }
            let nn: usize = std::env::var("LE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(32); let mut cur = ix[0] as u32; let t1 = std::time::Instant::now(); let mut out = vec![];
            for _ in 0..nn { out.push(cur); logits = gp.forward(cur); cur = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32; }
            println!("decode {} tok in {:.2}s = {:.1} tok/s  hits {} misses {}", nn, t1.elapsed().as_secs_f32(), nn as f32 / t1.elapsed().as_secs_f32(), gp.hits, gp.misses);
            println!("{}", String::from_utf8_lossy(&tk.decode_bytes(&out)));
        }
        cli::Cmd::H2d { path } => gpu::h2d_bench(&path),
        cli::Cmd::Serve(a) => {
            if let Err(e) = server::run(a).await {
                eprintln!("lastengine: {e}");
                std::process::exit(1);
            }
        }
    }
}
