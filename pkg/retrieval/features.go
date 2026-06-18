// SPDX-License-Identifier: AGPL-3.0-or-later
package retrieval

import (
	"math"
	"strings"
	"time"
	"unicode"
)

const AuxFeatureDim = 17
const LegacyAuxFeatureDim = 6

type AuxFeatureContext struct {
	QuestionType       string
	Now                time.Time
	LastQueryAt        time.Time
	NumPairs           int
	SignupAt           time.Time
	QueryEmbedding     []float32
	UserCorpusCentroid []float32
}

const (
	qtypeInfoExtractionIdx      = 6
	qtypeMultiSessionIdx        = 7
	qtypeKnowledgeUpdateIdx     = 8
	qtypeTemporalReasoningIdx   = 9
	qtypeAbstentionIdx          = 10
	auxHourSinIdx               = 11
	auxHourCosIdx               = 12
	auxLogSecsSinceLastQueryIdx = 13
	auxLogNumPairsIdx           = 14
	auxLogDaysSinceSignupIdx    = 15
	auxQueryCorpusDriftIdx      = 16
)

var temporalKeywords = []string{
	"yesterday", "today", "tonight", "this morning", "this afternoon",
	"last night", "last week", "last month", "last year",
	"recently", "lately", "just now", "earlier", "before",
	"ago", "previous", "prior", "past",
	"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
	"hour", "minute", "day", "week", "month",
}

var frequencyKeywords = []string{
	"often", "always", "usually", "frequently", "regularly",
	"keep talking", "keep discussing", "keep mentioning",
	"repeatedly", "constantly", "continuously",
	"common", "typical", "normal", "standard",
	"pattern", "habit", "routine", "recurring",
	"again and again", "over and over", "all the time",
	"every time", "each time",
}

var specificityKeywords = []string{
	"about", "regarding", "concerning", "related to",
	"specifically", "exactly", "precisely",
	"what is", "what are", "how does", "how do",
	"explain", "describe", "tell me about",
}

func ExtractAuxiliaryFeatures(query string) [AuxFeatureDim]float32 {
	return ExtractAuxiliaryFeaturesContext(query, AuxFeatureContext{})
}

func ExtractAuxiliaryFeaturesContext(query string, fctx AuxFeatureContext) [AuxFeatureDim]float32 {
	var out [AuxFeatureDim]float32
	words := strings.Fields(query)
	wordCount := len(words)
	normalizedWordCount := float32(wordCount) / 15.0
	if normalizedWordCount > 1.0 {
		normalizedWordCount = 1.0
	}
	out[0] = normalizedWordCount

	lower := strings.ToLower(query)
	temporalCount := countKeywordMatches(lower, temporalKeywords)
	out[1] = boolToFloat(temporalCount > 0)
	out[3] = minFloat32(float32(temporalCount)/3.0, 1.0)

	freqCount := countKeywordMatches(lower, frequencyKeywords)
	out[2] = boolToFloat(freqCount > 0)
	out[4] = minFloat32(float32(freqCount)/3.0, 1.0)

	specCount := countKeywordMatches(lower, specificityKeywords)
	out[5] = boolToFloat(specCount > 0 || hasNamedEntityPattern(words))

	if idx := questionTypeOneHotIdx(fctx.QuestionType); idx >= 0 {
		out[idx] = 1.0
	}
	if !fctx.Now.IsZero() {
		hour := float64(fctx.Now.Hour()) + float64(fctx.Now.Minute())/60.0
		angle := 2 * math.Pi * hour / 24.0
		out[auxHourSinIdx] = float32(math.Sin(angle))
		out[auxHourCosIdx] = float32(math.Cos(angle))
		if !fctx.LastQueryAt.IsZero() {
			delta := fctx.Now.Sub(fctx.LastQueryAt).Seconds()
			if delta < 0 {
				delta = 0
			}
			out[auxLogSecsSinceLastQueryIdx] = float32(math.Log1p(delta))
		}
	}
	if fctx.NumPairs > 0 {
		out[auxLogNumPairsIdx] = float32(math.Log1p(float64(fctx.NumPairs)))
	}
	if !fctx.SignupAt.IsZero() && !fctx.Now.IsZero() {
		days := fctx.Now.Sub(fctx.SignupAt).Hours() / 24.0
		if days < 0 {
			days = 0
		}
		out[auxLogDaysSinceSignupIdx] = float32(math.Log1p(days))
	}
	if drift, ok := cosineSimilarity(fctx.QueryEmbedding, fctx.UserCorpusCentroid); ok {
		out[auxQueryCorpusDriftIdx] = drift
	}
	return out
}

func PredictedIntent(w1, w2, w3 float64) string {
	if w1 >= w2 && w1 >= w3 {
		return "semantic"
	}
	if w2 >= w1 && w2 >= w3 {
		return "temporal"
	}
	return "frequency"
}

func questionTypeOneHotIdx(qtype string) int {
	if qtype == "" {
		return -1
	}
	q := strings.ToLower(strings.TrimSpace(qtype))
	if strings.HasSuffix(q, "_abs") || q == "abstention" {
		return qtypeAbstentionIdx
	}
	switch {
	case strings.Contains(q, "multi"):
		return qtypeMultiSessionIdx
	case strings.Contains(q, "knowledge"):
		return qtypeKnowledgeUpdateIdx
	case strings.Contains(q, "temporal"):
		return qtypeTemporalReasoningIdx
	case strings.Contains(q, "single") || strings.Contains(q, "extraction") || strings.Contains(q, "preference") || strings.Contains(q, "user") || strings.Contains(q, "assistant"):
		return qtypeInfoExtractionIdx
	}
	return -1
}

func cosineSimilarity(a, b []float32) (float32, bool) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, false
	}
	var dot, na, nb float64
	for i, x := range a {
		y := float64(b[i])
		fx := float64(x)
		dot += fx * y
		na += fx * fx
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0, false
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb))), true
}

func countKeywordMatches(text string, keywords []string) int {
	count := 0
	for _, kw := range keywords {
		if strings.Contains(text, kw) {
			count++
		}
	}
	return count
}

func hasNamedEntityPattern(words []string) bool {
	for i := 1; i < len(words); i++ {
		prev := words[i-1]
		if len(prev) > 0 {
			last := prev[len(prev)-1]
			if last == '.' || last == '!' || last == '?' {
				continue
			}
		}
		word := words[i]
		if len(word) > 1 {
			runes := []rune(word)
			if unicode.IsUpper(runes[0]) && unicode.IsLower(runes[1]) {
				return true
			}
		}
	}
	return false
}

func boolToFloat(b bool) float32 {
	if b {
		return 1.0
	}
	return 0.0
}

func minFloat32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
