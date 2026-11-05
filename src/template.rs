//! Chat template rendering (Jinja from GGUF `tokenizer.chat_template` or a file) via minijinja.
use minijinja::{context, Environment, Error, ErrorKind};
use serde_json::Value;

pub fn render(template: &str, messages: &[Value], tools: Option<&Value>, add_generation_prompt: bool,
              bos: &str, eos: &str) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |m: String| -> Result<String, Error> { Err(Error::new(ErrorKind::InvalidOperation, m)) });
    env.add_function("strftime_now", |_f: String| "2026-01-01".to_string());
    env.add_template("chat", template).map_err(|e| e.to_string())?;
    env.get_template("chat").unwrap()
        .render(context! { messages => minijinja::Value::from(minijinja::value::Serde(messages)), tools => minijinja::Value::from(minijinja::value::Serde(tools)), add_generation_prompt => add_generation_prompt,
                           bos_token => bos, eos_token => eos, enable_thinking => true })
        .map_err(|e| format!("{e:#}"))
}
