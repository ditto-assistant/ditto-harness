// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"strings"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

type HashEmbedder struct{}

func (HashEmbedder) Embed(_ context.Context, req harness.EmbedRequest) (harness.EmbedResponse, error) {
	out := make([][]float32, len(req.Texts))
	for i, text := range req.Texts {
		out[i] = hashEmbedding(text)
	}
	return harness.EmbedResponse{Embeddings: out}, nil
}

func hashEmbedding(text string) []float32 {
	vec := make([]float32, 768)
	for _, token := range strings.Fields(strings.ToLower(text)) {
		sum := sha256.Sum256([]byte(token))
		idx := int(binary.BigEndian.Uint16(sum[:2]) % uint16(len(vec)))
		vec[idx] += 1
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v * v)
	}
	if norm == 0 {
		vec[0] = 1
		return vec
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range vec {
		vec[i] *= scale
	}
	return vec
}
