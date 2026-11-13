//! GGUF v2/v3 reader, written from the public format description (ggml docs/gguf.md).
//! Layout: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count, kv pairs, tensor infos,
//! padding to `general.alignment` (default 32), then tensor data. Tensor offsets are relative to the data start.
use std::{collections::BTreeMap, fs::File, io::{self, BufReader, Read}, path::Path};

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool),
    Str(String), Arr(Vec<Value>), U64(u64), I64(i64), F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64, Value::U16(v) => *v as u64, Value::U32(v) => *v as u64, Value::U64(v) => *v,
            Value::Bool(b) => *b as u64,
            Value::I8(v) if *v >= 0 => *v as u64, Value::I16(v) if *v >= 0 => *v as u64,
            Value::I32(v) if *v >= 0 => *v as u64, Value::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_f32(&self) -> Option<f32> { match self { Value::F32(v) => Some(*v), Value::F64(v) => Some(*v as f32), _ => None } }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset relative to the start of the data section.
    pub offset: u64,
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where tensor data begins.
    pub data_start: u64,
}

struct R<T: Read> { r: T, pos: u64 }

impl<T: Read> R<T> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u8(&mut self) -> io::Result<u8> { Ok(self.bytes::<1>()?[0]) }
    fn u16(&mut self) -> io::Result<u16> { Ok(u16::from_le_bytes(self.bytes()?)) }
    fn u32(&mut self) -> io::Result<u32> { Ok(u32::from_le_bytes(self.bytes()?)) }
    fn u64(&mut self) -> io::Result<u64> { Ok(u64::from_le_bytes(self.bytes()?)) }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        if n > 1 << 28 { return Err(bad("string too long")); }
        let mut v = vec![0u8; n];
        self.r.read_exact(&mut v)?;
        self.pos += n as u64;
        String::from_utf8(v).map_err(|_| bad("invalid utf-8"))
    }
    fn value(&mut self, ty: u32) -> io::Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 30 { return Err(bad("array too long")); }
                let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
                for _ in 0..n { v.push(self.value(et)?); }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(bad("unknown value type")),
        })
    }
}

fn bad(m: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, m.to_string()) }

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Gguf> {
        let mut r = R { r: BufReader::with_capacity(1 << 20, File::open(path)?), pos: 0 };
        if &r.bytes::<4>()? != b"GGUF" { return Err(bad("not a GGUF file")); }
        let version = r.u32()?;
        if !(2..=3).contains(&version) { return Err(bad("unsupported GGUF version")); }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        if n_tensors > 1 << 24 || n_kv > 1 << 24 { return Err(bad("implausible counts")); }
        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let k = r.string()?;
            let t = r.u32()?;
            kv.insert(k, r.value(t)?);
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()?;
            if nd > 8 { return Err(bad("too many dims")); }
            let dims = (0..nd).map(|_| r.u64()).collect::<io::Result<Vec<_>>>()?;
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo { name, dims, ggml_type, offset });
        }
        let align = kv.get("general.alignment").and_then(Value::as_u64).unwrap_or(32).max(1);
        let data_start = r.pos.div_ceil(align) * align;
        Ok(Gguf { version, kv, tensors, data_start })
    }

    pub fn arch(&self) -> Option<&str> { self.kv.get("general.architecture")?.as_str() }
}
