// SPDX-License-Identifier: MIT
//! Loadable learned-weight MLP predictor.
//!
//! Binary artifact format (all little-endian):
//! - `u16` tensor count
//! - per tensor: `u16` name length, name bytes (UTF-8), `u8` ndims,
//!   `u32` per dim, then `product(dims)` f32 values.
//!
//! Required tensors: `embed_fc1.{weight,bias}`, `embed_ln1.{weight,bias}`,
//! `embed_fc2.{weight,bias}`, `embed_ln2.{weight,bias}`,
//! `fusion_fc.{weight,bias}`, `fusion_ln.{weight,bias}`,
//! `output_fc.{weight,bias}`.
//!
//! Derived config: `aux_dim = fusion_fc.weight.dims[1] - 64` (error if
//! negative; `LEGACY_AUX_FEATURE_DIM` when dims missing), `output_dim =
//! output_fc.weight.dims[0]` (3 when missing), `use_scale` when an
//! `output_scale.weight` tensor exists or `output_dim == V2_NUM_WEIGHTS + 1`.
//!
//! Forward pass (`predict_raw`): 768 -> fc1(256) -> layernorm -> relu ->
//! fc2(64) -> layernorm -> relu -> concat aux[..aux_dim] -> fusion(32) ->
//! layernorm -> relu -> output(output_dim). With `use_scale`, the last output
//! element is the scale (0 -> 1.0) and softmax applies to the head; otherwise
//! softmax applies to the whole output and scale is 1.0.
//!
//! # Aux features populated at inference
//!
//! At inference the harness fills only `Features::query`,
//! `Features::query_embedding`, `Features::now`, and
//! `Features::current_session_id` (and the MLP's aux extraction consumes only
//! the first three). Of the 17-dim aux layout ([`AUX_FEATURE_DIM`]), that
//! leaves the question-type one-hot (indexes 6-10), last-query recency (13),
//! num-pairs (14), signup age (15), and query-corpus drift (16) always zero.
//! A custom predictor trained against the 17-dim layout must reproduce that
//! skew or its training and inference distributions will not match.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use async_trait::async_trait;

use super::features::{extract_auxiliary_features_context, AuxFeatureContext, AUX_FEATURE_DIM};
use super::{
    default_weights, predicted_intent, Features, Result, WeightPredictor, Weights,
    LEGACY_AUX_FEATURE_DIM, V2_NUM_WEIGHTS, V2_WEIGHT_COSINE, V2_WEIGHT_NEIGHBOR_DENSITY,
    V2_WEIGHT_RECENCY_EXP, V2_WEIGHT_RECENCY_LINEAR, V2_WEIGHT_SESSION_CONTINUITY,
    V2_WEIGHT_SUBJECT_FREQUENCY, V2_WEIGHT_SUBJECT_SEM_MATCH,
};
use crate::types::Error;

const EMBED_INPUT_DIM: usize = 768;
const EMBED_HIDDEN1_DIM: usize = 256;
const EMBED_HIDDEN2_DIM: usize = 64;
const FUSION_DIM: usize = 32;

/// Hard cap on a single tensor's byte length in a model artifact. The real
/// model's largest tensor (`embed_fc1.weight`, 256x768 f32) is under 1 MiB,
/// so 64 MiB is generous headroom while refusing absurd/corrupt headers
/// before allocating.
const MAX_TENSOR_BYTES: usize = 64 * 1024 * 1024;

/// Fallback fusion weights returned by `predict_v2` when no model artifact is
/// loaded (cosine / recencyLinear / subjectFrequency). Deliberately NOT the
/// same values as [`default_weights`] (0.65 / 0.20 / 0.15): this is the
/// historical V2-path fallback, and changing it would silently shift ranking
/// for deployments running without a model artifact.
const UNLOADED_FALLBACK_COSINE: f64 = 0.6;
const UNLOADED_FALLBACK_RECENCY_LINEAR: f64 = 0.25;
const UNLOADED_FALLBACK_SUBJECT_FREQUENCY: f64 = 0.15;

/// Learned-weight MLP predictor. An unloaded/default
/// predictor returns [`default_weights`] from `predict`.
#[derive(Debug, Clone, Default)]
pub struct MlpPredictor {
    embed_fc1_w: Vec<f32>,
    embed_fc1_b: Vec<f32>,
    embed_ln1_w: Vec<f32>,
    embed_ln1_b: Vec<f32>,
    embed_fc2_w: Vec<f32>,
    embed_fc2_b: Vec<f32>,
    embed_ln2_w: Vec<f32>,
    embed_ln2_b: Vec<f32>,
    fusion_fc_w: Vec<f32>,
    fusion_fc_b: Vec<f32>,
    fusion_ln_w: Vec<f32>,
    fusion_ln_b: Vec<f32>,
    output_fc_w: Vec<f32>,
    output_fc_b: Vec<f32>,
    aux_dim: usize,
    output_dim: usize,
    use_scale: bool,
    loaded: bool,
}

impl MlpPredictor {
    /// Loads a model artifact from a file.
    pub fn load(path: &Path) -> Result<MlpPredictor> {
        let file = std::fs::File::open(path)?;
        MlpPredictor::load_from_reader(std::io::BufReader::new(file))
    }

    /// Loads a model artifact from a reader.
    pub fn load_from_reader(mut reader: impl Read) -> Result<MlpPredictor> {
        let num_tensors = read_u16(&mut reader, "tensor count")?;
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(num_tensors as usize);
        for _ in 0..num_tensors {
            let name_len = read_u16(&mut reader, "name len")? as usize;
            let mut name_buf = vec![0u8; name_len];
            reader
                .read_exact(&mut name_buf)
                .map_err(|err| Error::Other(format!("read name: {err}")))?;
            let name = String::from_utf8(name_buf)
                .map_err(|err| Error::Other(format!("read name: {err}")))?;
            let ndims = read_u8(&mut reader, "ndims")?;
            let mut dims = Vec::with_capacity(ndims as usize);
            let mut total: usize = 1;
            for _ in 0..ndims {
                let dim = read_u32(&mut reader, "dim")?;
                dims.push(dim);
                total = total
                    .checked_mul(dim as usize)
                    .ok_or_else(|| Error::Other(format!("tensor {name}: dimension overflow")))?;
            }
            let byte_len = total
                .checked_mul(4)
                .ok_or_else(|| Error::Other(format!("tensor {name}: byte length overflow")))?;
            if byte_len > MAX_TENSOR_BYTES {
                return Err(Error::Other(format!(
                    "tensor {name}: {byte_len} bytes exceeds the {MAX_TENSOR_BYTES}-byte cap"
                )));
            }
            let mut buf = vec![0u8; byte_len];
            reader
                .read_exact(&mut buf)
                .map_err(|err| Error::Other(format!("read data for {name}: {err}")))?;
            let data = crate::db::decode_f32_blob(&buf);
            tensors.insert(name, Tensor { dims, data });
        }

        let mut p = MlpPredictor::default();
        // Derived config first, read from the tensor map before the data is
        // moved out.
        match tensors.get("fusion_fc.weight") {
            Some(t) if t.dims.len() == 2 => {
                let input_dim = t.dims[1] as usize;
                if input_dim < EMBED_HIDDEN2_DIM {
                    return Err(Error::Other(format!(
                        "invalid fusion_fc input dim {input_dim}"
                    )));
                }
                p.aux_dim = input_dim - EMBED_HIDDEN2_DIM;
            }
            _ => p.aux_dim = LEGACY_AUX_FEATURE_DIM,
        }
        match tensors.get("output_fc.weight") {
            Some(t) if t.dims.len() == 2 => p.output_dim = t.dims[0] as usize,
            _ => p.output_dim = 3,
        }
        p.use_scale =
            tensors.contains_key("output_scale.weight") || p.output_dim == V2_NUM_WEIGHTS + 1;

        let fused_dim = EMBED_HIDDEN2_DIM + p.aux_dim;
        // (name, destination, expected length) — lengths are validated up
        // front: malformed artifacts must error at load time, not panic in
        // predict_raw.
        let required: [(&str, &mut Vec<f32>, usize); 14] = [
            (
                "embed_fc1.weight",
                &mut p.embed_fc1_w,
                EMBED_HIDDEN1_DIM * EMBED_INPUT_DIM,
            ),
            ("embed_fc1.bias", &mut p.embed_fc1_b, EMBED_HIDDEN1_DIM),
            ("embed_ln1.weight", &mut p.embed_ln1_w, EMBED_HIDDEN1_DIM),
            ("embed_ln1.bias", &mut p.embed_ln1_b, EMBED_HIDDEN1_DIM),
            (
                "embed_fc2.weight",
                &mut p.embed_fc2_w,
                EMBED_HIDDEN2_DIM * EMBED_HIDDEN1_DIM,
            ),
            ("embed_fc2.bias", &mut p.embed_fc2_b, EMBED_HIDDEN2_DIM),
            ("embed_ln2.weight", &mut p.embed_ln2_w, EMBED_HIDDEN2_DIM),
            ("embed_ln2.bias", &mut p.embed_ln2_b, EMBED_HIDDEN2_DIM),
            (
                "fusion_fc.weight",
                &mut p.fusion_fc_w,
                FUSION_DIM * fused_dim,
            ),
            ("fusion_fc.bias", &mut p.fusion_fc_b, FUSION_DIM),
            ("fusion_ln.weight", &mut p.fusion_ln_w, FUSION_DIM),
            ("fusion_ln.bias", &mut p.fusion_ln_b, FUSION_DIM),
            (
                "output_fc.weight",
                &mut p.output_fc_w,
                p.output_dim * FUSION_DIM,
            ),
            ("output_fc.bias", &mut p.output_fc_b, p.output_dim),
        ];
        for (name, dest, want_len) in required {
            let tensor = tensors
                .get(name)
                .ok_or_else(|| Error::Other(format!("missing tensor: {name}")))?;
            if tensor.data.len() != want_len {
                return Err(Error::Other(format!(
                    "tensor {name}: {} values, want {want_len}",
                    tensor.data.len()
                )));
            }
            dest.clone_from(&tensor.data);
        }
        p.loaded = true;
        Ok(p)
    }

    /// True once a model artifact has been loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Auxiliary feature dimension expected by the loaded model.
    pub fn aux_dim(&self) -> usize {
        self.aux_dim
    }

    /// Output dimension of the loaded model.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Raw V2 prediction: returns (weights, scale, intent).
    /// An unloaded predictor returns the fallback (0.6 / 0.25 / 0.15 over
    /// `V2_NUM_WEIGHTS` slots, scale 1.0, "semantic"). Scale 0 is coerced to
    /// 1.0. Intent is `predicted_intent(weights[0], weights[1], weights[2])`.
    pub fn predict_v2(
        &self,
        embedding: &[f32],
        aux_features: &[f32; AUX_FEATURE_DIM],
    ) -> (Vec<f64>, f64, &'static str) {
        if !self.is_loaded() {
            let mut weights = vec![0f64; V2_NUM_WEIGHTS];
            weights[V2_WEIGHT_COSINE] = UNLOADED_FALLBACK_COSINE;
            weights[V2_WEIGHT_RECENCY_LINEAR] = UNLOADED_FALLBACK_RECENCY_LINEAR;
            weights[V2_WEIGHT_SUBJECT_FREQUENCY] = UNLOADED_FALLBACK_SUBJECT_FREQUENCY;
            return (weights, 1.0, "semantic");
        }
        let (raw, scale_f) = self.predict_raw(embedding, aux_features);
        let weights: Vec<f64> = raw.iter().map(|v| f64::from(*v)).collect();
        let mut scale = f64::from(scale_f);
        if scale == 0.0 {
            scale = 1.0;
        }
        let mut intent = "semantic";
        if weights.len() >= 3 {
            intent = predicted_intent(weights[0], weights[1], weights[2]);
        }
        (weights, scale, intent)
    }

    /// The forward pass. Short query embeddings are zero-padded to 768 —
    /// this function must stay panic-free on any input.
    fn predict_raw(
        &self,
        embedding: &[f32],
        aux_features: &[f32; AUX_FEATURE_DIM],
    ) -> (Vec<f32>, f32) {
        let mut input = vec![0f32; EMBED_INPUT_DIM];
        let n = embedding.len().min(EMBED_INPUT_DIM);
        input[..n].copy_from_slice(&embedding[..n]);

        let mut h1 = linear_forward(
            &self.embed_fc1_w,
            &self.embed_fc1_b,
            &input,
            EMBED_HIDDEN1_DIM,
            EMBED_INPUT_DIM,
        );
        layer_norm(&mut h1, &self.embed_ln1_w, &self.embed_ln1_b);
        relu(&mut h1);
        let mut h2 = linear_forward(
            &self.embed_fc2_w,
            &self.embed_fc2_b,
            &h1,
            EMBED_HIDDEN2_DIM,
            EMBED_HIDDEN1_DIM,
        );
        layer_norm(&mut h2, &self.embed_ln2_w, &self.embed_ln2_b);
        relu(&mut h2);

        let aux_dim = if self.aux_dim == 0 {
            LEGACY_AUX_FEATURE_DIM
        } else {
            self.aux_dim
        };
        let fused_len = EMBED_HIDDEN2_DIM + aux_dim;
        let mut fused = vec![0f32; fused_len];
        fused[..EMBED_HIDDEN2_DIM].copy_from_slice(&h2);
        let aux_copy = aux_dim.min(AUX_FEATURE_DIM);
        fused[EMBED_HIDDEN2_DIM..EMBED_HIDDEN2_DIM + aux_copy]
            .copy_from_slice(&aux_features[..aux_copy]);
        let mut h3 = linear_forward(
            &self.fusion_fc_w,
            &self.fusion_fc_b,
            &fused,
            FUSION_DIM,
            fused_len,
        );
        layer_norm(&mut h3, &self.fusion_ln_w, &self.fusion_ln_b);
        relu(&mut h3);

        let output_dim = if self.output_dim == 0 {
            3
        } else {
            self.output_dim
        };
        let mut out = linear_forward(
            &self.output_fc_w,
            &self.output_fc_b,
            &h3,
            output_dim,
            FUSION_DIM,
        );
        if self.use_scale && output_dim > 1 {
            let mut scale = out[output_dim - 1];
            if scale == 0.0 {
                scale = 1.0;
            }
            out.truncate(output_dim - 1);
            softmax(&mut out);
            return (out, scale);
        }
        softmax(&mut out);
        (out, 1.0)
    }
}

#[async_trait]
impl WeightPredictor for MlpPredictor {
    /// Unloaded -> defaults; otherwise extract aux features from
    /// `features.query` (+ now + query embedding), run `predict_v2`, and map
    /// via [`weights_from_slice`].
    async fn predict(&self, features: &Features) -> Result<Weights> {
        if !self.is_loaded() {
            return Ok(default_weights());
        }
        let aux = extract_auxiliary_features_context(
            &features.query,
            &AuxFeatureContext {
                now: features.now,
                query_embedding: features.query_embedding.clone(),
                ..AuxFeatureContext::default()
            },
        );
        let (weights, scale, _) = self.predict_v2(&features.query_embedding, &aux);
        Ok(weights_from_slice(&weights, scale))
    }
}

/// Maps a raw weight slice + scale onto [`Weights`].
/// Missing entries keep their [`default_weights`] values; scale 0 -> 1.
pub fn weights_from_slice(weights: &[f64], scale: f64) -> Weights {
    let mut out = default_weights();
    if let Some(v) = weights.get(V2_WEIGHT_COSINE) {
        out.cosine = *v;
    }
    if let Some(v) = weights.get(V2_WEIGHT_RECENCY_LINEAR) {
        out.recency_linear = *v;
    }
    if let Some(v) = weights.get(V2_WEIGHT_RECENCY_EXP) {
        out.recency_exp = *v;
    }
    if let Some(v) = weights.get(V2_WEIGHT_SUBJECT_FREQUENCY) {
        out.subject_frequency = *v;
    }
    if let Some(v) = weights.get(V2_WEIGHT_SUBJECT_SEM_MATCH) {
        out.subject_sem_match = *v;
    }
    if let Some(v) = weights.get(V2_WEIGHT_SESSION_CONTINUITY) {
        out.session_continuity = *v;
    }
    if let Some(v) = weights.get(V2_WEIGHT_NEIGHBOR_DENSITY) {
        out.neighbor_density = *v;
    }
    out.scale = if scale == 0.0 { 1.0 } else { scale };
    out
}

struct Tensor {
    dims: Vec<u32>,
    data: Vec<f32>,
}

fn read_u8(reader: &mut impl Read, what: &str) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader
        .read_exact(&mut buf)
        .map_err(|err| Error::Other(format!("read {what}: {err}")))?;
    Ok(buf[0])
}

fn read_u16(reader: &mut impl Read, what: &str) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader
        .read_exact(&mut buf)
        .map_err(|err| Error::Other(format!("read {what}: {err}")))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(reader: &mut impl Read, what: &str) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|err| Error::Other(format!("read {what}: {err}")))?;
    Ok(u32::from_le_bytes(buf))
}

/// `y = W x + b` with row-major `W[out_dim][in_dim]`.
fn linear_forward(w: &[f32], b: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; out_dim];
    for (i, yi) in y.iter_mut().enumerate() {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut sum = 0f32;
        for (wj, xj) in row.iter().zip(x.iter()) {
            sum += wj * xj;
        }
        *yi = sum + b[i];
    }
    y
}

/// In-place layer normalization with gamma/beta.
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let variance: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = (1.0 / (f64::from(variance) + 1e-5).sqrt()) as f32;
    for (i, v) in x.iter_mut().enumerate() {
        *v = gamma[i] * (*v - mean) * inv_std + beta[i];
    }
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// In-place numerically stable softmax.
fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().copied().fold(x[0], f32::max);
    let mut sum = 0f64;
    for v in x.iter_mut() {
        *v = f64::from(*v - max_val).exp() as f32;
        sum += f64::from(*v);
    }
    if sum == 0.0 {
        return;
    }
    for v in x.iter_mut() {
        *v = (f64::from(*v) / sum) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use chrono::Utc;

    /// Builds a minimal valid model artifact with constant-filled tensors.
    fn write_mock_model(aux_dim: usize, output_dim: usize, with_scale_tensor: bool) -> Vec<u8> {
        struct T {
            name: &'static str,
            dims: Vec<u32>,
        }
        let mut tensors = vec![
            T {
                name: "embed_fc1.weight",
                dims: vec![256, 768],
            },
            T {
                name: "embed_fc1.bias",
                dims: vec![256],
            },
            T {
                name: "embed_ln1.weight",
                dims: vec![256],
            },
            T {
                name: "embed_ln1.bias",
                dims: vec![256],
            },
            T {
                name: "embed_fc2.weight",
                dims: vec![64, 256],
            },
            T {
                name: "embed_fc2.bias",
                dims: vec![64],
            },
            T {
                name: "embed_ln2.weight",
                dims: vec![64],
            },
            T {
                name: "embed_ln2.bias",
                dims: vec![64],
            },
            T {
                name: "fusion_fc.weight",
                dims: vec![32, (64 + aux_dim) as u32],
            },
            T {
                name: "fusion_fc.bias",
                dims: vec![32],
            },
            T {
                name: "fusion_ln.weight",
                dims: vec![32],
            },
            T {
                name: "fusion_ln.bias",
                dims: vec![32],
            },
            T {
                name: "output_fc.weight",
                dims: vec![output_dim as u32, 32],
            },
            T {
                name: "output_fc.bias",
                dims: vec![output_dim as u32],
            },
        ];
        if with_scale_tensor {
            tensors.push(T {
                name: "output_scale.weight",
                dims: vec![1],
            });
        }
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(tensors.len() as u16).to_le_bytes());
        for t in &tensors {
            buf.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
            buf.extend_from_slice(t.name.as_bytes());
            buf.push(t.dims.len() as u8);
            let mut size = 1usize;
            for d in &t.dims {
                buf.extend_from_slice(&d.to_le_bytes());
                size *= *d as usize;
            }
            let fill_value: f32 = match t.name {
                "embed_ln1.weight" | "embed_ln2.weight" | "fusion_ln.weight" => 1.0,
                "embed_fc1.weight" => 1.0 / (768f32).sqrt(),
                "embed_fc2.weight" => 1.0 / (256f32).sqrt(),
                "fusion_fc.weight" => 1.0 / ((64 + aux_dim) as f32).sqrt(),
                "output_fc.weight" => 0.1,
                _ => 0.0,
            };
            for _ in 0..size {
                buf.extend_from_slice(&fill_value.to_le_bytes());
            }
        }
        buf
    }

    /// A V2-shaped artifact loads with the expected aux/output dims.
    #[test]
    fn load_v2_shape() {
        let model = write_mock_model(AUX_FEATURE_DIM, V2_NUM_WEIGHTS + 1, true);
        let p = MlpPredictor::load_from_reader(model.as_slice()).expect("load");
        assert!(p.is_loaded());
        assert_eq!(p.aux_dim(), AUX_FEATURE_DIM);
        assert_eq!(p.output_dim(), V2_NUM_WEIGHTS + 1);
    }

    /// `predict` through the `WeightPredictor` trait populates weights.
    #[tokio::test]
    async fn predict_implements_weight_predictor() {
        let model = write_mock_model(AUX_FEATURE_DIM, V2_NUM_WEIGHTS + 1, true);
        let p = MlpPredictor::load_from_reader(model.as_slice()).expect("load");
        let predictor: &dyn WeightPredictor = &p;
        let embedding: Vec<f32> = (0..768).map(|i| 0.001 * i as f32).collect();
        let weights = predictor
            .predict(&Features {
                query: "recently repeated retrieval pattern".to_string(),
                now: Some(
                    Utc.with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
                        .single()
                        .expect("timestamp"),
                ),
                query_embedding: embedding,
                ..Features::default()
            })
            .await
            .expect("predict");
        assert!(
            weights.cosine != 0.0 && weights.scale != 0.0,
            "weights not populated: {weights:?}"
        );
    }

    #[test]
    fn unloaded_predictor_falls_back() {
        let p = MlpPredictor::default();
        assert!(!p.is_loaded());
        let (weights, scale, intent) = p.predict_v2(&[], &[0f32; AUX_FEATURE_DIM]);
        assert_eq!(weights.len(), V2_NUM_WEIGHTS);
        assert_eq!(weights[V2_WEIGHT_COSINE], 0.6);
        assert_eq!(weights[V2_WEIGHT_RECENCY_LINEAR], 0.25);
        assert_eq!(weights[V2_WEIGHT_SUBJECT_FREQUENCY], 0.15);
        assert_eq!(scale, 1.0);
        assert_eq!(intent, "semantic");
    }

    #[test]
    fn missing_tensor_errors() {
        // Truncate the model so a required tensor never appears.
        let err = MlpPredictor::load_from_reader([1u8, 0u8].as_slice()).expect_err("error");
        assert!(err.to_string().contains("read"), "{err}");
    }

    #[test]
    fn weights_from_slice_fills_and_defaults() {
        let w = weights_from_slice(&[0.5, 0.3], 0.0);
        assert_eq!(w.cosine, 0.5);
        assert_eq!(w.recency_linear, 0.3);
        // Missing entries keep default_weights() values.
        assert_eq!(w.subject_frequency, default_weights().subject_frequency);
        assert_eq!(w.scale, 1.0, "scale 0 coerces to 1");

        let full = weights_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 2.0);
        assert_eq!(full.recency_exp, 0.3);
        assert_eq!(full.subject_sem_match, 0.5);
        assert_eq!(full.session_continuity, 0.6);
        assert_eq!(full.neighbor_density, 0.7);
        assert_eq!(full.scale, 2.0);
    }
}
