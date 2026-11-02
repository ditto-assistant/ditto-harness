// Upgrade check for @ditto/harness-node: a database written by one build must
// open, read back and accept writes under another. Used when bumping turso.
//
// 1. With the old binding: node upgrade-check.mjs write <db> <fingerprint.json>
// 2. With the new binding: node upgrade-check.mjs check <db> <fingerprint.json>
//
// Uses the deterministic "hash" embedder so both runs rank identically.

import { readFileSync, writeFileSync } from "node:fs";
import assert from "node:assert/strict";
import { Harness, harnessVersion } from "./index.js";

const [mode, dbPath, fingerprintPath] = process.argv.slice(2);
if (!["write", "check"].includes(mode) || !dbPath || !fingerprintPath) {
  console.error("usage: node upgrade-check.mjs write|check <db> <fingerprint.json>");
  process.exit(2);
}

const UID = "upgrade-user";
const TOPICS = ["hiking", "sourdough", "chess", "rust", "violin", "garden", "kayak", "tax"];
const QUERIES = TOPICS.map((t) => `notes about ${t} number 3`);

// minSimilarity 0 means "default" (0.15); -1 admits every stored memory, so
// one unbounded search enumerates the user's whole store.
async function allMemoryIds(harness) {
  const all = await harness.searchMemories(UID, "upgrade", { minSimilarity: -1, limit: 100000 });
  return all.map((m) => m.id).sort();
}

async function fingerprint(harness) {
  const all = await allMemoryIds(harness);
  const searches = {};
  for (const q of QUERIES) {
    const hits = await harness.searchMemories(UID, q, { minSimilarity: -1, limit: 10 });
    searches[q] = hits.map((m) => [m.id, m.title || m.summary || ""]);
  }
  const subjects = {};
  for (const t of TOPICS) {
    subjects[t] = (await harness.searchSubjects(UID, t)).map((s) => s.text).sort();
  }
  return { all, searches, subjects };
}

const harness = await Harness.open(dbPath, { embedder: "hash" });
console.log(`harness version: ${harnessVersion()} (${mode})`);

if (mode === "write") {
  await harness.seedUser(UID);
  let n = 0;
  for (const topic of TOPICS) {
    for (let i = 0; i < 25; i++) {
      await harness.saveMemory({
        userId: UID,
        id: `pair-${topic}-${i}`,
        prompt: `Question ${i} about ${topic}: what should I know about ${topic} number ${i}?`,
        response: `Answer ${i}: ${topic} detail ${i} with some longer explanation text.`,
        summary: `${topic} notes number ${i}`,
        sessionId: i % 2 ? "personal" : "work",
        daysAgo: i,
        subjects: [{ text: `${topic} topic`, description: `All about ${topic}`, key: i === 0 }],
      });
      n++;
    }
  }
  const fp = await fingerprint(harness);
  writeFileSync(fingerprintPath, JSON.stringify({ memories: n, ...fp }, null, 2));
  console.log(`wrote ${n} memories and fingerprint`);
} else {
  const expected = JSON.parse(readFileSync(fingerprintPath, "utf8"));
  const actual = await fingerprint(harness);
  // A fingerprint of empty results would match an empty store; refuse it.
  assert.equal(expected.all.length, expected.memories, "fingerprint does not list every memory");
  let hits = 0;
  for (const q of QUERIES) hits += expected.searches[q].length;
  const subjectHits = Object.values(expected.subjects).reduce((n, s) => n + s.length, 0);
  assert.ok(hits > 0 && subjectHits > 0, "fingerprint has no search results; it proves nothing");
  assert.deepEqual(actual.all, expected.all, "stored memories changed after upgrade");
  assert.deepEqual(actual.subjects, expected.subjects, "subject search changed after upgrade");
  assert.deepEqual(actual.searches, expected.searches, "memory search changed after upgrade");
  // The upgraded build must still write to the old file.
  await harness.seedUser(UID);
  const saved = await harness.saveMemory({
    userId: UID,
    id: "pair-after-upgrade",
    prompt: "Written after the upgrade: quokka facts",
    response: "Quokkas live on Rottnest Island.",
    summary: "quokka facts after upgrade",
    sessionId: "personal",
    daysAgo: 0,
    subjects: [{ text: "quokka topic", description: "post-upgrade write", key: true }],
  });
  assert.equal(saved.id, "pair-after-upgrade");
  const after = await harness.searchMemories(UID, "quokka facts after upgrade", { minSimilarity: -1 });
  assert.ok(after.some((m) => m.id === "pair-after-upgrade"), "post-upgrade write not searchable");
  console.log(
    `upgrade check passed: all ${actual.all.length} memories present, ${QUERIES.length} memory queries (${hits} hits) and ${TOPICS.length} subject queries (${subjectHits} hits) identical; post-upgrade write searchable`
  );
}
