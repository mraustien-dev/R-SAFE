// Local database: officers enrolled on this phone, duty sessions, and the upload queue.
// The app has no screen that edits or deletes queued evidence.
import * as SQLite from 'expo-sqlite';

let dbPromise = null;

export function getDb() {
  if (!dbPromise) dbPromise = open();
  return dbPromise;
}

async function open() {
  const db = await SQLite.openDatabaseAsync('rsafe.db');
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY NOT NULL, value TEXT);
    CREATE TABLE IF NOT EXISTS officers (
      id TEXT PRIMARY KEY NOT NULL,
      email TEXT,
      full_name TEXT,
      pin_salt TEXT NOT NULL,
      pin_hash TEXT NOT NULL,
      failed INTEGER NOT NULL DEFAULT 0,
      enrolled_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS duty (
      id TEXT PRIMARY KEY NOT NULL,
      officer_id TEXT NOT NULL,
      started_at INTEGER NOT NULL,
      ended_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS outbox (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      kind TEXT NOT NULL,              -- 'duty' | 'violation'
      seq INTEGER,                     -- violation number on this phone
      body TEXT NOT NULL,              -- exact signed text
      signature TEXT,                  -- record signature (violations)
      files TEXT,                      -- JSON {frame, rider, plate} local paths
      folder TEXT,                     -- evidence folder
      created_at INTEGER NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      last_error TEXT,
      state TEXT NOT NULL DEFAULT 'pending', -- pending | sent | rejected
      case_no TEXT,
      sent_at INTEGER
    );
    CREATE INDEX IF NOT EXISTS outbox_state ON outbox (state, id);
  `);
  return db;
}

export async function kvGet(key) {
  const db = await getDb();
  const row = await db.getFirstAsync('SELECT value FROM kv WHERE key = ?', key);
  return row ? row.value : null;
}

export async function kvSet(key, value) {
  const db = await getDb();
  await db.runAsync('INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)', key, value == null ? null : String(value));
}

/** Next violation number on this phone (1, 2, 3...). Gaps show up on the dashboard. */
export async function nextSeq() {
  const db = await getDb();
  let seq = 0;
  await db.withExclusiveTransactionAsync(async (tx) => {
    const row = await tx.getFirstAsync("SELECT value FROM kv WHERE key = 'last_seq'");
    seq = (row ? Number(row.value) : 0) + 1;
    await tx.runAsync("INSERT OR REPLACE INTO kv (key, value) VALUES ('last_seq', ?)", String(seq));
  });
  return seq;
}

/** Violation rows in one upload state ('pending' | 'sent' | 'rejected'), newest first. Read only. */
export async function listOutbox(state, limit = 50, offset = 0) {
  const db = await getDb();
  return db.getAllAsync(
    `SELECT id, seq, state, body, files, attempts, last_error, created_at, sent_at, case_no
     FROM outbox WHERE kind = 'violation' AND state = ? ORDER BY id DESC LIMIT ? OFFSET ?`,
    state, limit, offset,
  );
}

export async function queueCounts() {
  const db = await getDb();
  const rows = await db.getAllAsync(
    "SELECT state, COUNT(*) AS n FROM outbox WHERE kind = 'violation' GROUP BY state",
  );
  const out = { pending: 0, sent: 0, rejected: 0 };
  rows.forEach((r) => { out[r.state] = r.n; });
  return out;
}
