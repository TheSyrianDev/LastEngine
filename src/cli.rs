//! Command line. Flag names match what existing llama-swap configs pass; unset flags are chosen by auto-tune (milestone 7).
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lastengine", version)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    Serve(ServeArgs),
    /// Print GGUF header metadata.
    Inspect { path: String },
    H2d { path: String },
    /// GPU forward: top-N logprobs + 32-token greedy decode.
    Gpu { path: String, prompt: String, #[arg(default_value_t = 5)] top: usize, #[arg(default_value_t = 128)] margin: usize },
    /// CPU reference: print top-N next-token logprobs for PROMPT.
    Ref { path: String, prompt: String, #[arg(default_value_t = 5)] top: usize },
    Dequant { path: String, name: String, n: usize, out: String },
    Template { path: String },
    /// Print token ids for TEXT using the model's tokenizer.
    Tokenize { path: String, text: String },
}

#[derive(clap::Args, Clone, Debug)]
pub struct ServeArgs {
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub addr: String,
    #[arg(long)]
    pub gpu: bool,
    #[arg(long)]
    pub ctx: Option<usize>,
    #[arg(long)]
    pub kv: Option<String>,
    #[arg(long)]
    pub chunk: Option<usize>,
    #[arg(long = "vram-margin")]
    pub vram_margin: Option<usize>,
    #[arg(long)]
    pub threads: Option<usize>,
    #[arg(long)]
    pub mtp: Option<usize>,
    #[arg(long = "slot-save-path")]
    pub slot_save_path: Option<String>,
    #[arg(long = "chat-template-file")]
    pub chat_template_file: Option<String>,
    #[arg(long = "rope-scaling")]
    pub rope_scaling: Option<String>,
    #[arg(long = "rope-scale")]
    pub rope_scale: Option<f32>,
    #[arg(long = "yarn-orig-ctx")]
    pub yarn_orig_ctx: Option<usize>,
    /// Use the built-in stub backend (no model needed); for protocol tests.
    #[arg(long)]
    pub stub: bool,
}
