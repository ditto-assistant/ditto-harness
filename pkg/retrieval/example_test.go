// SPDX-License-Identifier: AGPL-3.0-or-later
package retrieval_test

import (
	"context"
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/retrieval"
)

func ExampleStaticPredictor() {
	predictor := retrieval.StaticPredictor{Weights: retrieval.Weights{
		Cosine:           0.7,
		RecencyLinear:    0.2,
		SubjectFrequency: 0.1,
		Scale:            1,
	}}
	weights, err := predictor.Predict(context.Background(), retrieval.Features{
		Query: "recent project preferences",
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(weights.Cosine)
	// Output:
	// 0.7
}

func ExampleLoadMLPPredictor() {
	// Host applications can load their deployed model artifact and pass the
	// predictor into memory.NewStore. The artifact is intentionally not bundled
	// with ditto-harness.
	_, _ = retrieval.LoadMLPPredictor("/opt/ditto/retrieval-model.bin")
}
