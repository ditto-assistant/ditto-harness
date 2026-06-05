package loopdetect

const MaxConsecutiveSameToolCalls = 3

type ToolCall struct {
	Name string
	Args string
}

type Detection struct {
	Turn             int
	Call             ToolCall
	ConsecutiveCalls int
}

type Detector struct {
	consecutiveCount int
	lastCall         ToolCall
}

func (d *Detector) RecordTurn(turn int, calls []ToolCall) (Detection, bool) {
	if len(calls) == 0 {
		d.reset()
		return Detection{}, false
	}
	if len(calls) != 1 {
		d.reset()
		return Detection{}, false
	}
	call := calls[0]
	if call.Name == d.lastCall.Name && call.Args == d.lastCall.Args {
		d.consecutiveCount++
	} else {
		d.lastCall = call
		d.consecutiveCount = 1
	}
	if d.consecutiveCount >= MaxConsecutiveSameToolCalls {
		return Detection{Turn: turn, Call: d.lastCall, ConsecutiveCalls: d.consecutiveCount}, true
	}
	return Detection{}, false
}

func (d *Detector) reset() {
	d.consecutiveCount = 0
	d.lastCall = ToolCall{}
}
