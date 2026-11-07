//! Chat template rendering (Jinja from GGUF `tokenizer.chat_template` or a file) via minijinja.
use minijinja::{context, Environment, Error, ErrorKind};
use serde_json::Value;

pub fn render(template: &str, messages: &[Value], tools: Option<&Value>, add_generation_prompt: bool,
              bos: &str, eos: &str, enable_thinking: bool) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |m: String| -> Result<String, Error> { Err(Error::new(ErrorKind::InvalidOperation, m)) });
    // HF-style tojson: keyword args (ensure_ascii, indent, sort_keys) are accepted; no HTML escaping of < > & '.
    env.add_filter("tojson", |v: minijinja::Value, kw: minijinja::value::Kwargs| -> Result<minijinja::Value, Error> {
        let indent: Option<usize> = kw.get("indent")?;
        let _: Option<bool> = kw.get("ensure_ascii")?; let _: Option<bool> = kw.get("sort_keys")?;
        kw.assert_all_used()?;
        let s = match indent {
            Some(n) => { let ind = " ".repeat(n); let mut buf = Vec::new(); let f = serde_json::ser::PrettyFormatter::with_indent(ind.as_bytes());
                         serde::Serialize::serialize(&v, &mut serde_json::Serializer::with_formatter(&mut buf, f)).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
                         String::from_utf8(buf).unwrap_or_default() }
            None => serde_json::to_string(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?,
        };
        Ok(minijinja::Value::from_safe_string(s))
    });
    env.add_function("strftime_now", |_f: String| "2026-01-01".to_string());
    env.add_template("chat", template).map_err(|e| e.to_string())?;
    env.get_template("chat").unwrap()
        .render(context! { messages => minijinja::Value::from(minijinja::value::Serde(messages)), tools => minijinja::Value::from(minijinja::value::Serde(tools)), add_generation_prompt => add_generation_prompt,
                           bos_token => bos, eos_token => eos, enable_thinking => enable_thinking })
        .map_err(|e| format!("{e:#}"))
}
