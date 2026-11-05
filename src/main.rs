//! lastengine: `lastengine serve --model X.gguf --addr 127.0.0.1:PORT [flags]`
mod backend;
mod cli;
mod server;

use clap::Parser;

#[tokio::main]
async fn main() {
    let cli::Cli { cmd } = cli::Cli::parse();
    match cmd {
        cli::Cmd::Serve(a) => {
            if let Err(e) = server::run(a).await {
                eprintln!("lastengine: {e}");
                std::process::exit(1);
            }
        }
    }
}
