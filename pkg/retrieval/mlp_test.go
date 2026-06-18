// SPDX-License-Identifier: AGPL-3.0-or-later
package retrieval

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"testing"
	"time"
)

func writeMockModel(t *testing.T, auxDim, outputDim int, withScaleTensor bool) []byte {
	t.Helper()
	type tensor struct {
		name string
		dims []uint32
	}
	tensors := []tensor{
		{"embed_fc1.weight", []uint32{256, 768}},
		{"embed_fc1.bias", []uint32{256}},
		{"embed_ln1.weight", []uint32{256}},
		{"embed_ln1.bias", []uint32{256}},
		{"embed_fc2.weight", []uint32{64, 256}},
		{"embed_fc2.bias", []uint32{64}},
		{"embed_ln2.weight", []uint32{64}},
		{"embed_ln2.bias", []uint32{64}},
		{"fusion_fc.weight", []uint32{32, uint32(64 + auxDim)}},
		{"fusion_fc.bias", []uint32{32}},
		{"fusion_ln.weight", []uint32{32}},
		{"fusion_ln.bias", []uint32{32}},
		{"output_fc.weight", []uint32{uint32(outputDim), 32}},
		{"output_fc.bias", []uint32{uint32(outputDim)}},
	}
	if withScaleTensor {
		tensors = append(tensors, tensor{"output_scale.weight", []uint32{1}})
	}
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint16(len(tensors))); err != nil {
		t.Fatalf("write tensor count: %v", err)
	}
	for _, ten := range tensors {
		if err := binary.Write(&buf, binary.LittleEndian, uint16(len(ten.name))); err != nil {
			t.Fatalf("write name len: %v", err)
		}
		buf.Write([]byte(ten.name))
		if err := binary.Write(&buf, binary.LittleEndian, uint8(len(ten.dims))); err != nil {
			t.Fatalf("write ndims: %v", err)
		}
		size := 1
		for _, d := range ten.dims {
			if err := binary.Write(&buf, binary.LittleEndian, d); err != nil {
				t.Fatalf("write dim: %v", err)
			}
			size *= int(d)
		}
		data := make([]float32, size)
		switch ten.name {
		case "embed_ln1.weight", "embed_ln2.weight", "fusion_ln.weight":
			for i := range data {
				data[i] = 1
			}
		case "embed_fc1.weight":
			fill(data, float32(1.0/math.Sqrt(768)))
		case "embed_fc2.weight":
			fill(data, float32(1.0/math.Sqrt(256)))
		case "fusion_fc.weight":
			fill(data, float32(1.0/math.Sqrt(float64(64+auxDim))))
		case "output_fc.weight":
			fill(data, 0.1)
		}
		if err := binary.Write(&buf, binary.LittleEndian, data); err != nil {
			t.Fatalf("write data: %v", err)
		}
	}
	return buf.Bytes()
}

func TestMLPPredictorLoadV2Shape(t *testing.T) {
	p, err := LoadMLPPredictorFromReader(bytes.NewReader(writeMockModel(t, AuxFeatureDim, V2NumWeights+1, true)))
	if err != nil {
		t.Fatalf("LoadMLPPredictorFromReader: %v", err)
	}
	if !p.IsLoaded() {
		t.Fatal("predictor should be loaded")
	}
	if p.AuxDim() != AuxFeatureDim {
		t.Fatalf("AuxDim = %d, want %d", p.AuxDim(), AuxFeatureDim)
	}
	if p.OutputDim() != V2NumWeights+1 {
		t.Fatalf("OutputDim = %d, want %d", p.OutputDim(), V2NumWeights+1)
	}
}

func TestMLPPredictorPredictImplementsWeightPredictor(t *testing.T) {
	p, err := LoadMLPPredictorFromReader(bytes.NewReader(writeMockModel(t, AuxFeatureDim, V2NumWeights+1, true)))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var predictor WeightPredictor = p
	embedding := make([]float32, 768)
	for i := range embedding {
		embedding[i] = 0.001 * float32(i)
	}
	weights, err := predictor.Predict(context.Background(), Features{
		Query:          "recently repeated retrieval pattern",
		Now:            time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC),
		QueryEmbedding: embedding,
	})
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if weights.Cosine == 0 || weights.Scale == 0 {
		t.Fatalf("weights not populated: %+v", weights)
	}
}

func TestExtractAuxiliaryFeaturesContext(t *testing.T) {
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	got := ExtractAuxiliaryFeaturesContext("what did Peyton mention recently", AuxFeatureContext{
		QuestionType:       "temporal_reasoning",
		Now:                now,
		LastQueryAt:        now.Add(-time.Hour),
		NumPairs:           10,
		SignupAt:           now.Add(-24 * time.Hour),
		QueryEmbedding:     []float32{1, 0},
		UserCorpusCentroid: []float32{1, 0},
	})
	if got[0] == 0 || got[1] == 0 || got[qtypeTemporalReasoningIdx] != 1 {
		t.Fatalf("expected query length, temporal, and qtype features: %v", got)
	}
	if got[auxQueryCorpusDriftIdx] != 1 {
		t.Fatalf("query corpus drift = %f, want 1", got[auxQueryCorpusDriftIdx])
	}
}

func fill(xs []float32, v float32) {
	for i := range xs {
		xs[i] = v
	}
}
