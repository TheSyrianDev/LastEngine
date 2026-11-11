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
    /// Print hardware facts as stable JSON on stdout (for launchers / model recommendation).
    /// --json is accepted for launchers; output is always JSON.
    Probe { #[arg(long)] json: bool },
    /// Measure candidate kv/chunk settings on this machine and cache the best for `serve`.
    Tune { model: String, #[arg(long, default_value_t = 8192)] ctx: usize },
    /// Print GGUF header metadata.
    Inspect { path: String },
    H2d { path: String },
    /// GPU forward: top-N logprobs + 32-token greedy decode.
    Gpu { path: String, prompt: String, #[arg(default_value_t = 5)] top: usize, #[arg(default_value_t = 128)] margin: usize },
    /// CPU reference: print top-N next-token logprobs for PROMPT.
    /// Clover block-diffusion draft quality check on the CPU reference (draft a block, compare with greedy AR)
    CloverDraft { path: String, prompt: String, #[arg(default_value_t = 1)] blocks: usize },
    /// Dense model: CPU greedy for N tokens, GPU teacher-forced on the same tokens; report first logit/argmax divergences
    DenseCmp { path: String, prompt: String, #[arg(default_value_t = 100)] n: usize },
    /// Greedy-decode N tokens on the GPU and print JSON {prompt_ids,gen_ids,top} (top = per step top-5 [id,logprob]) for teacher-forced comparison with another engine
    TfDump { path: String, prompt: String, #[arg(default_value_t = 100)] n: usize },
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
