//! lastengine: `lastengine serve --model X.gguf --addr 127.0.0.1:PORT [flags]`
mod backend;
mod cli;
mod probe;
mod tune;
mod cpu_ref;
mod dense;
mod gguf;
mod gpu;
mod local;
mod server;
mod quant;
mod quant_ik;
#[path = "quant_ik_tables.rs"]
mod quant_ik_tables;
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
            println!("{}", template::render(t, &msgs, None, true, "", "", true).unwrap_or_else(|e| e));
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
        cli::Cmd::Ref { path, prompt, top } if gguf::Gguf::open(&path).ok().and_then(|g| dense::Dense::supports(&g)).is_some() => {
            let g = gguf::Gguf::open(&path).expect("open");
            let tk = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            let m = dense::Dense::load(&path, &g);
            let mut ids = tk.encode(&prompt);
            if g.kv.get("tokenizer.ggml.add_bos_token").map_or(false, |v| matches!(v, gguf::Value::Bool(true))) { if let Some(b) = tk.bos { ids.insert(0, b); } }
            let mut st = m.new_state();
            let t0 = std::time::Instant::now();
            let mut logits = vec![];
            for (i, id) in ids.iter().enumerate() { logits = m.forward(&mut st, *id, i); }
            eprintln!("{} tok in {:.2}s", ids.len(), t0.elapsed().as_secs_f32());
            let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
            let mut ix: Vec<usize> = (0..logits.len()).collect();
            ix.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
            for i in ix.iter().take(top) { println!("{} {:?} logprob {:.4}", i, tk.decode_bytes(&[*i as u32]).iter().map(|b| *b as char).collect::<String>(), logits[*i] - lse); }
        }
        cli::Cmd::CloverDraft { path, prompt, blocks } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let tk = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            let m = cpu_ref::Model::load(&path, &g);
            let ids = tk.encode(&prompt);
            let mut st = m.new_state();
            let mut logits = vec![];
            for (i, id) in ids.iter().enumerate() { logits = m.forward(&mut st, *id, i); }
            let mut pos = ids.len();
            let argmax = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::MIN), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0 as u32;
            let mask = 151669u32;
            let bs = 32usize;
            let mut gen: Vec<u32> = vec![];
            for b in 0..blocks {
                // AR reference for the block, from a copy of the state
                let mut st2 = st.clone();
                let mut l2 = logits.clone();
                let mut ar = vec![];
                for j in 0..bs { let t = argmax(&l2); ar.push(t); l2 = m.forward(&mut st2, t, pos + j); }
                // draft: iterative denoising
                let mut blk = vec![mask; bs];
                let t0 = std::time::Instant::now();
                let mut iters = 0;
                while blk.iter().any(|t| *t == mask) {
                    iters += 1;
                    let lg = m.preview(&st, &blk, pos);
                    // logits at position i predict token i (same-position logits); mask id is never emitted
                    let mut best: Option<(usize, u32, f32)> = None;
                    let mut taken = 0;
                    let mut cand = vec![];
                    for i in 0..bs {
                        if blk[i] != mask { continue; }
                        let mut l = lg[i].clone(); l[mask as usize] = f32::NEG_INFINITY;
                        let a = argmax(&l);
                        let mx = l[a as usize];
                        let conf = 1.0 / l.iter().map(|v| (v - mx).exp()).sum::<f32>();
                        cand.push((i, a, conf));
                        if best.map_or(true, |b| conf > b.2) { best = Some((i, a, conf)); }
                    }
                    for &(i, a, conf) in &cand { if conf > 0.95 { blk[i] = a; taken += 1; } }
                    if taken == 0 { let (i, a, _) = best.unwrap(); blk[i] = a; }
                    eprintln!("  iter {iters}: {} pending", blk.iter().filter(|t| **t == mask).count());
                }
                let acc = blk.iter().zip(&ar).take_while(|(a, b)| a == b).count();
                let same = blk.iter().zip(&ar).filter(|(a, b)| a == b).count();
                println!("block {b}: {iters} denoise iters in {:.1}s; prefix match {acc}/{bs}, positional match {same}/{bs}", t0.elapsed().as_secs_f32());
                println!("  draft: {:?}", tk.decode_bytes(&blk).iter().map(|b| *b as char).collect::<String>());
                println!("  AR   : {:?}", tk.decode_bytes(&ar).iter().map(|b| *b as char).collect::<String>());
                for j in 0..bs { logits = m.forward(&mut st, ar[j], pos + j); }
                pos += bs; gen.extend(&ar);
            }
        }
        cli::Cmd::DenseCmp { path, prompt, n } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let tk = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            let cpu = dense::Dense::load(&path, &g);
            let mut gp = gpu::dense::DenseGpu::new(dense::Dense::load(&path, &g), 4096);
            let mut st = cpu.new_state();
            let ids = tk.encode(&prompt);
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let lp = |v: &[f32], i: usize| { let mx = v.iter().cloned().fold(f32::MIN, f32::max); v[i] - mx - v.iter().map(|a| (a - mx).exp()).sum::<f32>().ln() };
            let (mut lc, mut lg) = (vec![], vec![]);
            for (i, t) in ids.iter().enumerate() { lc = cpu.forward(&mut st, *t, i); }
            let t0 = std::time::Instant::now(); lg = gp.prefill(&ids); println!("gpu prefill {} tok in {:.3}s", ids.len(), t0.elapsed().as_secs_f32());
            let (mut diff, mut agree, mut maxd, mut sumd) = (0, 0, 0f32, 0f32);
            for k in 0..n {
                let (a, b) = (am(&lc), am(&lg));
                let d = (lp(&lc, a) - lp(&lg, a)).abs(); maxd = maxd.max(d); sumd += d;
                if a == b { agree += 1 } else if diff < 8 { diff += 1; println!("step {k}: cpu {a} lp {:.3} (gpu lp {:.3}) vs gpu {b} lp {:.3} (cpu lp {:.3})", lp(&lc, a), lp(&lg, a), lp(&lg, b), lp(&lc, b)); }
                let t = a as u32; lc = cpu.forward(&mut st, t, ids.len() + k); lg = gp.forward(t);
            }
            println!("{agree}/{n} argmax agree, mean |dlogprob| {:.4}, max {:.4}", sumd / n as f32, maxd);
        }
        cli::Cmd::TfDump { path, prompt, n } => {
            let g = gguf::Gguf::open(&path).expect("open");
            let tk = tokenizer::Tokenizer::from_gguf(&g).expect("tokenizer");
            if let Err(e) = dense::check_arch(&g) { eprintln!("lastengine: {e}"); std::process::exit(1); }
            let ids = tk.encode(&prompt);
            let dense_kind = dense::Dense::supports(&g).is_some();
            let mut step: Box<dyn FnMut(Option<u32>) -> Vec<f32>> = if dense_kind {
                let mut gp = gpu::dense::DenseGpu::new(dense::Dense::load(&path, &g), 4096);
                let ids2 = ids.clone(); let mut first = true;
                Box::new(move |t| if first { first = false; gp.prefill(&ids2) } else { gp.forward(t.unwrap()) })
            } else {
                let m = std::sync::Arc::new(cpu_ref::Model::load(&path, &g));
                let mut gp = gpu::Gpu::new(m, 4096, 128);
                let ids2 = ids.clone(); let mut first = true;
                Box::new(move |t| if first { first = false; gp.prefill(&ids2) } else { gp.forward(t.unwrap()) })
            };
            let mut lg = step(None); let mut gen = vec![]; let mut top = vec![];
            for _ in 0..n {
                let mx = lg.iter().cloned().fold(f32::MIN, f32::max); let lse = mx + lg.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
                let mut ix: Vec<usize> = (0..lg.len()).collect(); ix.sort_by(|a, b| lg[*b].partial_cmp(&lg[*a]).unwrap());
                top.push(ix.iter().take(5).map(|i| (*i, lg[*i] - lse)).collect::<Vec<_>>());
                gen.push(ix[0] as u32); lg = step(Some(ix[0] as u32));
            }
            println!("{}", serde_json::json!({"prompt_ids": ids, "gen_ids": gen, "top": top}));
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
            if gp.draft_on {
                let (mut blocks, mut acc) = (0, 0);
                let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
                while out.len() < nn {
                    let d = gp.clover_draft(cur, 0.95);
                    gp.snapshot_state(false); let p0 = gp.pos;
                    let mut toks = vec![cur]; toks.extend_from_slice(&d);
                    let rows = gp.feed_rows(&toks);
                    let mut m = 0; let mut next = am(&rows[0]);
                    while m < d.len() && next == d[m] { m += 1; next = am(&rows[m]); }
                    out.extend_from_slice(&toks[..=m]);
                    if m + 1 < toks.len() { gp.snapshot_state(true); gp.pos = p0; gp.feed(&toks[..=m]); }
                    blocks += 1; acc += m; cur = next;
                }
                println!("draft-self: {blocks} blocks, {} denoise passes, {acc} drafted tokens accepted ({:.1}/block)", gp.drafts_iters, acc as f32 / blocks as f32);
            } else if gp.mtp_enabled() {
                let (mut steps, mut acc) = (0, 0);
                while out.len() < nn { let (t, n) = gp.spec_step(cur); steps += 1; acc += t.len() - 1; out.extend(t); cur = n; }
                println!("mtp: {steps} steps, {acc} drafts accepted ({:.0}%)", 100.0 * acc as f32 / steps as f32);
            } else {
            for _ in 0..nn { out.push(cur); logits = gp.forward(cur); cur = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32; }
            }
            let nn = out.len();
            println!("decode {} tok in {:.2}s = {:.1} tok/s  hits {} misses {}", nn, t1.elapsed().as_secs_f32(), nn as f32 / t1.elapsed().as_secs_f32(), gp.hits, gp.misses);
            println!("{}", String::from_utf8_lossy(&tk.decode_bytes(&out)));
        }
        cli::Cmd::Tune { model, ctx } => tune::run(&model, ctx),
        cli::Cmd::Probe { .. } => println!("{}", probe::probe_json()),
        cli::Cmd::H2d { path } => gpu::h2d_bench(&path),
        cli::Cmd::Serve(a) => {
            if let Err(e) = server::run(a).await {
                eprintln!("lastengine: {e}");
                std::process::exit(1);
            }
        }
    }
}
