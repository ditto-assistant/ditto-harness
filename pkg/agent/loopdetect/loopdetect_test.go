package loopdetect

import "testing"

func TestDetectorDetectsRepeatedSingleToolCall(t *testing.T) {
	var detector Detector
	call := []ToolCall{{Name: "search_memories", Args: `{"queries":["x"]}`}}
	if _, ok := detector.RecordTurn(0, call); ok {
		t.Fatal("first turn detected loop")
	}
	if _, ok := detector.RecordTurn(1, call); ok {
		t.Fatal("second turn detected loop")
	}
	detection, ok := detector.RecordTurn(2, call)
	if !ok {
		t.Fatal("third repeated turn did not detect loop")
	}
	if detection.ConsecutiveCalls != MaxConsecutiveSameToolCalls {
		t.Fatalf("consecutive calls = %d, want %d", detection.ConsecutiveCalls, MaxConsecutiveSameToolCalls)
	}
}

func TestDetectorResetsOnMultiToolTurn(t *testing.T) {
	var detector Detector
	call := []ToolCall{{Name: "search_memories", Args: `{}`}}
	detector.RecordTurn(0, call)
	detector.RecordTurn(1, call)
	if _, ok := detector.RecordTurn(2, []ToolCall{{Name: "a"}, {Name: "b"}}); ok {
		t.Fatal("multi-tool turn should reset")
	}
	if _, ok := detector.RecordTurn(3, call); ok {
		t.Fatal("single turn after reset detected loop")
	}
}
