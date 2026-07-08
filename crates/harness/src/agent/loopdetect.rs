// SPDX-License-Identifier: MIT
//! Detection of repeated identical tool calls.

/// Number of identical consecutive calls that triggers detection.
pub const MAX_CONSECUTIVE_SAME_TOOL_CALLS: usize = 3;

/// A tool call identity: name plus raw argument string.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCallKey {
    pub name: String,
    pub args: String,
}

/// A detected loop.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Detection {
    pub turn: usize,
    pub call: ToolCallKey,
    pub consecutive_calls: usize,
}

/// Stateful detector fed one turn at a time.
#[derive(Debug, Clone, Default)]
pub struct Detector {
    consecutive_count: usize,
    last_call: ToolCallKey,
}

impl Detector {
    /// Records a turn's tool calls. Zero or multiple calls reset the
    /// detector. A single call increments the streak when identical
    /// (name + args) to the previous call, else restarts it at 1. Returns
    /// `Some(Detection)` once the streak reaches
    /// [`MAX_CONSECUTIVE_SAME_TOOL_CALLS`].
    pub fn record_turn(&mut self, turn: usize, calls: &[ToolCallKey]) -> Option<Detection> {
        if calls.len() != 1 {
            self.reset();
            return None;
        }
        let call = &calls[0];
        if *call == self.last_call {
            self.consecutive_count += 1;
        } else {
            self.last_call = call.clone();
            self.consecutive_count = 1;
        }
        if self.consecutive_count >= MAX_CONSECUTIVE_SAME_TOOL_CALLS {
            return Some(Detection {
                turn,
                call: self.last_call.clone(),
                consecutive_calls: self.consecutive_count,
            });
        }
        None
    }

    fn reset(&mut self) {
        self.consecutive_count = 0;
        self.last_call = ToolCallKey::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call() -> Vec<ToolCallKey> {
        vec![ToolCallKey {
            name: "search_memories".to_string(),
            args: r#"{"queries":["x"]}"#.to_string(),
        }]
    }

    #[test]
    fn detector_detects_repeated_single_tool_call() {
        let mut detector = Detector::default();
        assert!(
            detector.record_turn(0, &call()).is_none(),
            "first turn detected loop"
        );
        assert!(
            detector.record_turn(1, &call()).is_none(),
            "second turn detected loop"
        );
        let detection = detector
            .record_turn(2, &call())
            .expect("third repeated turn did not detect loop");
        assert_eq!(detection.consecutive_calls, MAX_CONSECUTIVE_SAME_TOOL_CALLS);
        assert_eq!(detection.turn, 2);
    }

    #[test]
    fn detector_resets_on_multi_tool_turn() {
        let mut detector = Detector::default();
        detector.record_turn(0, &call());
        detector.record_turn(1, &call());
        let multi = vec![
            ToolCallKey {
                name: "a".to_string(),
                args: String::new(),
            },
            ToolCallKey {
                name: "b".to_string(),
                args: String::new(),
            },
        ];
        assert!(
            detector.record_turn(2, &multi).is_none(),
            "multi-tool turn should reset"
        );
        assert!(
            detector.record_turn(3, &call()).is_none(),
            "single turn after reset detected loop"
        );
    }
}
