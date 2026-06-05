package cost

import "github.com/ditto-assistant/ditto-harness/pkg/harness"

type Collector struct {
	items []harness.CostedUsage
}

func (c *Collector) Add(item *harness.CostedUsage) {
	if item == nil {
		return
	}
	c.items = append(c.items, *item)
}

func (c *Collector) Items() []harness.CostedUsage {
	out := make([]harness.CostedUsage, len(c.items))
	copy(out, c.items)
	return out
}

func (c *Collector) Total() harness.Cost {
	var total harness.Cost
	for _, item := range c.items {
		if item.Cost.Currency != "" {
			total.Currency = item.Cost.Currency
		}
		total.Amount += item.Cost.Amount
	}
	return total
}
