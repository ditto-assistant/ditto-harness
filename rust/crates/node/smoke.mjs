// Smoke test for @ditto/harness-node.
//
// Opens a temp database, seeds a user with 2 memories, and verifies vector
// search returns the seeded memory. Uses the real Ollama embedder
// (embeddinggemma) when a local server is reachable, otherwise falls back to
// the deterministic "hash" stub embedder so the test passes offline.
//
// Run: node smoke.mjs

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import assert from "node:assert/strict";
import { Harness, harnessVersion } from "./index.js";

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
const EMBED_MODEL = "embeddinggemma";

/** Returns true when Ollama is reachable and has the embedding model pulled. */
async function ollamaAvailable() {
  try {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/tags`, {
      signal: AbortSignal.timeout(1500),
    });
    if (!res.ok) return false;
    const body = await res.json();
    return (body.models ?? []).some((m) => (m.name ?? "").startsWith(EMBED_MODEL));
  } catch {
    return false;
  }
}

const dir = mkdtempSync(join(tmpdir(), "ditto-harness-node-smoke-"));
const dbPath = join(dir, "smoke.db");
let failed = false;

try {
  console.log(`harness version: ${harnessVersion()}`);

  const useOllama = await ollamaAvailable();
  const embedder = useOllama ? "ollama" : "hash";
  console.log(`embedder: ${embedder}${useOllama ? ` (${EMBED_MODEL} @ ${OLLAMA_BASE_URL})` : " (offline stub)"}`);

  const harness = await Harness.open(dbPath, {
    ollamaBaseUrl: OLLAMA_BASE_URL,
    embedder,
  });

  const uid = "smoke-user";
  await harness.seedUser(uid);

  const saved1 = await harness.saveMemory({
    userId: uid,
    id: "smoke-pair-001",
    prompt:
      "I'm planning a three-day hiking trip around Mount Rainier in September with my sister.",
    response:
      "Late September is perfect for Rainier: golden larches and fewer crowds. The Spray Park to Mowich Lake loop fits three days well.",
    summary: "Planned a 3-day September Mount Rainier hiking trip with sister.",
    sessionId: "personal",
    daysAgo: 5,
    subjects: [
      {
        text: "Mount Rainier trip",
        description: "3-day September hiking trip with sister",
        key: true,
      },
    ],
  });
  assert.equal(saved1.id, "smoke-pair-001", "saveMemory must echo the stable pair id");
  assert.equal(saved1.userId, uid);
  console.log(`saved memory 1: ${saved1.id} (${saved1.title || saved1.summary})`);

  const saved2 = await harness.saveMemory({
    userId: uid,
    id: "smoke-pair-002",
    prompt: "My sourdough starter Ferris is bubbling. What's the feeding schedule?",
    response:
      "Feed Ferris 1:1:1 starter to flour to water by weight once a day at room temperature.",
    summary: "Sourdough starter feeding schedule for Ferris.",
    sessionId: "personal",
    daysAgo: 2,
    subjects: [
      { text: "sourdough baking", description: "Sourdough hobby; starter named Ferris", key: true },
    ],
  });
  assert.equal(saved2.id, "smoke-pair-002");
  console.log(`saved memory 2: ${saved2.id} (${saved2.title || saved2.summary})`);

  // searchMemories must surface the hiking memory for a hiking query.
  const hikingHits = await harness.searchMemories(
    uid,
    "hiking trip around Mount Rainier in September",
    { minSimilarity: 0.05 }
  );
  assert.ok(hikingHits.length >= 1, "searchMemories returned no results");
  assert.ok(
    hikingHits.some((m) => m.id === "smoke-pair-001"),
    `expected smoke-pair-001 in results, got: ${hikingHits.map((m) => m.id).join(", ")}`
  );
  console.log(
    `searchMemories ok: ${hikingHits.length} hit(s), top = ${hikingHits[0].id} (similarity ${hikingHits[0].similarity?.toFixed(3)})`
  );

  // Subject search must surface the seeded subject graph.
  const subjects = await harness.searchSubjects(uid, "sourdough starter baking");
  assert.ok(
    subjects.some((s) => s.text === "sourdough baking"),
    `expected "sourdough baking" subject, got: ${subjects.map((s) => s.text).join(", ")}`
  );
  console.log(`searchSubjects ok: ${subjects.map((s) => s.text).join(", ")}`);

  console.log("smoke test passed");
} catch (err) {
  failed = true;
  console.error("smoke test FAILED:", err);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
