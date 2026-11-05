//! lastengine: `lastengine serve --model X.gguf --addr 127.0.0.1:PORT [flags]`
mod backend;
mod cli;
mod gguf;
mod server;
mod tokenizer;

use clap::Parser;

#[tokio::main]
async fn main() {
    let cli::Cli { cmd } = cli::Cli::parse();
    match cmd {
        cli::Cmd::Inspect { path } => match gguf::Gguf::open(&path) {
            Ok(g) => {
                println!("GGUF v{} arch={:?} tensors={} data_start={}", g.version, g.arch(), g.tensors.len(), g.data_start);
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
        cli::Cmd::Serve(a) => {
            if let Err(e) = server::run(a).await {
                eprintln!("lastengine: {e}");
                std::process::exit(1);
            }
        }
    }
}
