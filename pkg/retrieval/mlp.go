// SPDX-License-Identifier: AGPL-3.0-or-later
package retrieval

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

type MLPPredictor struct {
	embedFC1W []float32
	embedFC1B []float32
	embedLN1W []float32
	embedLN1B []float32
	embedFC2W []float32
	embedFC2B []float32
	embedLN2W []float32
	embedLN2B []float32
	fusionFCW []float32
	fusionFCB []float32
	fusionLNW []float32
	fusionLNB []float32
	outputFCW []float32
	outputFCB []float32
	auxDim    int
	outputDim int
	useScale  bool
	loaded    bool
}

func LoadMLPPredictor(path string) (*MLPPredictor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return LoadMLPPredictorFromReader(f)
}

func LoadMLPPredictorFromReader(r io.Reader) (*MLPPredictor, error) {
	p := &MLPPredictor{}
	if err := p.load(r); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *MLPPredictor) IsLoaded() bool {
	return p != nil && p.loaded
}

func (p *MLPPredictor) AuxDim() int {
	if p == nil {
		return 0
	}
	return p.auxDim
}

func (p *MLPPredictor) OutputDim() int {
	if p == nil {
		return 0
	}
	return p.outputDim
}

func (p *MLPPredictor) Predict(_ context.Context, features Features) (Weights, error) {
	if !p.IsLoaded() {
		return DefaultWeights(), nil
	}
	aux := ExtractAuxiliaryFeaturesContext(features.Query, AuxFeatureContext{
		Now:            features.Now,
		QueryEmbedding: features.QueryEmbedding,
	})
	weights, scale, _ := p.PredictV2(features.QueryEmbedding, aux)
	return WeightsFromSlice(weights, scale), nil
}

func (p *MLPPredictor) PredictV2(embedding []float32, auxFeatures [AuxFeatureDim]float32) ([]float64, float64, string) {
	if !p.IsLoaded() {
		weights := make([]float64, V2NumWeights)
		weights[V2WeightCosine] = 0.6
		weights[V2WeightRecencyLinear] = 0.25
		weights[V2WeightSubjectFrequency] = 0.15
		return weights, 1.0, "semantic"
	}
	raw, scaleF := p.predictRaw(embedding, auxFeatures)
	weights := make([]float64, len(raw))
	for i, v := range raw {
		weights[i] = float64(v)
	}
	scale := float64(scaleF)
	if scale == 0 {
		scale = 1.0
	}
	intent := "semantic"
	if len(weights) >= 3 {
		intent = PredictedIntent(weights[0], weights[1], weights[2])
	}
	return weights, scale, intent
}

func WeightsFromSlice(weights []float64, scale float64) Weights {
	out := DefaultWeights()
	if len(weights) > V2WeightCosine {
		out.Cosine = weights[V2WeightCosine]
	}
	if len(weights) > V2WeightRecencyLinear {
		out.RecencyLinear = weights[V2WeightRecencyLinear]
	}
	if len(weights) > V2WeightRecencyExp {
		out.RecencyExp = weights[V2WeightRecencyExp]
	}
	if len(weights) > V2WeightSubjectFrequency {
		out.SubjectFrequency = weights[V2WeightSubjectFrequency]
	}
	if len(weights) > V2WeightSubjectSemMatch {
		out.SubjectSemMatch = weights[V2WeightSubjectSemMatch]
	}
	if len(weights) > V2WeightSessionContinuity {
		out.SessionContinuity = weights[V2WeightSessionContinuity]
	}
	if len(weights) > V2WeightNeighborDensity {
		out.NeighborDensity = weights[V2WeightNeighborDensity]
	}
	if scale == 0 {
		scale = 1
	}
	out.Scale = scale
	return out
}

func (p *MLPPredictor) load(r io.Reader) error {
	var numTensors uint16
	if err := binary.Read(r, binary.LittleEndian, &numTensors); err != nil {
		return fmt.Errorf("read tensor count: %w", err)
	}
	type tensorInfo struct {
		dims []uint32
		data []float32
	}
	tensors := make(map[string]tensorInfo, numTensors)
	for i := 0; i < int(numTensors); i++ {
		var nameLen uint16
		if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
			return fmt.Errorf("read name len: %w", err)
		}
		nameBuf := make([]byte, nameLen)
		if _, err := io.ReadFull(r, nameBuf); err != nil {
			return fmt.Errorf("read name: %w", err)
		}
		name := string(nameBuf)
		var ndims uint8
		if err := binary.Read(r, binary.LittleEndian, &ndims); err != nil {
			return fmt.Errorf("read ndims: %w", err)
		}
		dims := make([]uint32, ndims)
		totalSize := 1
		for j := 0; j < int(ndims); j++ {
			if err := binary.Read(r, binary.LittleEndian, &dims[j]); err != nil {
				return fmt.Errorf("read dim: %w", err)
			}
			totalSize *= int(dims[j])
		}
		data := make([]float32, totalSize)
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return fmt.Errorf("read data for %s: %w", name, err)
		}
		tensors[name] = tensorInfo{dims: dims, data: data}
	}
	required := map[string]*[]float32{
		"embed_fc1.weight": &p.embedFC1W,
		"embed_fc1.bias":   &p.embedFC1B,
		"embed_ln1.weight": &p.embedLN1W,
		"embed_ln1.bias":   &p.embedLN1B,
		"embed_fc2.weight": &p.embedFC2W,
		"embed_fc2.bias":   &p.embedFC2B,
		"embed_ln2.weight": &p.embedLN2W,
		"embed_ln2.bias":   &p.embedLN2B,
		"fusion_fc.weight": &p.fusionFCW,
		"fusion_fc.bias":   &p.fusionFCB,
		"fusion_ln.weight": &p.fusionLNW,
		"fusion_ln.bias":   &p.fusionLNB,
		"output_fc.weight": &p.outputFCW,
		"output_fc.bias":   &p.outputFCB,
	}
	for name, ptr := range required {
		t, ok := tensors[name]
		if !ok {
			return fmt.Errorf("missing tensor: %s", name)
		}
		*ptr = t.data
	}
	if t, ok := tensors["fusion_fc.weight"]; ok && len(t.dims) == 2 {
		p.auxDim = int(t.dims[1]) - 64
		if p.auxDim < 0 {
			return fmt.Errorf("invalid fusion_fc input dim %d", t.dims[1])
		}
	} else {
		p.auxDim = LegacyAuxFeatureDim
	}
	if t, ok := tensors["output_fc.weight"]; ok && len(t.dims) == 2 {
		p.outputDim = int(t.dims[0])
	} else {
		p.outputDim = 3
	}
	if _, ok := tensors["output_scale.weight"]; ok {
		p.useScale = true
	} else if p.outputDim == V2NumWeights+1 {
		p.useScale = true
	}
	p.loaded = true
	return nil
}

func (p *MLPPredictor) predictRaw(embedding []float32, auxFeatures [AuxFeatureDim]float32) ([]float32, float32) {
	h1 := linearForward(p.embedFC1W, p.embedFC1B, embedding, 256, 768)
	layerNorm(h1, p.embedLN1W, p.embedLN1B)
	relu(h1)
	h2 := linearForward(p.embedFC2W, p.embedFC2B, h1, 64, 256)
	layerNorm(h2, p.embedLN2W, p.embedLN2B)
	relu(h2)

	auxDim := p.auxDim
	if auxDim <= 0 {
		auxDim = LegacyAuxFeatureDim
	}
	fusedLen := 64 + auxDim
	fused := make([]float32, fusedLen)
	copy(fused, h2)
	if auxDim <= AuxFeatureDim {
		copy(fused[64:], auxFeatures[:auxDim])
	} else {
		copy(fused[64:64+AuxFeatureDim], auxFeatures[:])
	}
	h3 := linearForward(p.fusionFCW, p.fusionFCB, fused, 32, fusedLen)
	layerNorm(h3, p.fusionLNW, p.fusionLNB)
	relu(h3)

	outputDim := p.outputDim
	if outputDim <= 0 {
		outputDim = 3
	}
	out := linearForward(p.outputFCW, p.outputFCB, h3, outputDim, 32)
	if p.useScale && outputDim > 1 {
		head := out[:outputDim-1]
		softmax(head)
		scale := out[outputDim-1]
		if scale == 0 {
			scale = 1.0
		}
		return head, scale
	}
	softmax(out)
	return out, 1.0
}

func linearForward(W, b, x []float32, outDim, inDim int) []float32 {
	y := make([]float32, outDim)
	for i := 0; i < outDim; i++ {
		var sum float32
		row := W[i*inDim : (i+1)*inDim]
		for j := 0; j < inDim; j++ {
			sum += row[j] * x[j]
		}
		y[i] = sum + b[i]
	}
	return y
}

func layerNorm(x, gamma, beta []float32) {
	n := len(x)
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(n)
	var variance float32
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(n)
	invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-5))
	for i := range x {
		x[i] = gamma[i]*(x[i]-mean)*invStd + beta[i]
	}
}

func relu(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

func softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += float64(x[i])
	}
	if sum == 0 {
		return
	}
	for i := range x {
		x[i] = float32(float64(x[i]) / sum)
	}
}
