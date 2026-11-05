//! lastengine: `lastengine serve --model X.gguf --addr 127.0.0.1:PORT [flags]`
mod backend;
mod cli;
mod gguf;
mod server;
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
        cli::Cmd::Serve(a) => {
            if let Err(e) = server::run(a).await {
                eprintln!("lastengine: {e}");
                std::process::exit(1);
            }
        }
    }
}
