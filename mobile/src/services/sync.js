// Automatic upload. The officer has no button for this and cannot stop it:
// it runs whenever the phone gets internet, after every new violation, and every 30 s.
import * as Network from 'expo-network';
import RsafeDetector from '../../modules/rsafe-detector';
import { getDb, kvGet, kvSet, queueCounts } from './db';
import { ApiError, sendDuty, sendHeartbeat, sendViolation } from './api';
import { APP_VERSION } from './device';

const listeners = new Set();
let status = { online: false, syncing: false, blocked: null, lastSyncAt: null, counts: { pending: 0, sent: 0, rejected: 0 } };
let running = false;
let again = false;
let started = false;

export function onSyncStatus(fn) {
  listeners.add(fn);
  fn(status);
  return () => listeners.delete(fn);
}

function emit(patch) {
  status = { ...status, ...patch };
  listeners.forEach((fn) => fn(status));
}

// The server will never accept these, retrying cannot help: keep the row, stop retrying it.
const PERMANENT = new Set([
  'bad_record', 'bad_seq', 'bad_detected_at', 'missing_fields', 'bad_json', 'duty_rejected',
  'image_too_large:frame', 'image_too_large:rider', 'image_too_large:plate',
]);
// The whole phone is refused: stop until an admin fixes it.
const DEVICE_BLOCKED = new Set(['unknown_device', 'bad_signature', 'device_disabled']);

async function isOnline() {
  try {
    const s = await Network.getNetworkStateAsync();
    return !!s.isConnected && s.isInternetReachable !== false;
  } catch {
    return false;
  }
}

export async function syncNow() {
  if (running) { again = true; return; }
  running = true;
  try {
    do {
      again = false;
      await runOnce();
    } while (again);
  } finally {
    running = false;
  }
}

async function runOnce() {
  const online = await isOnline();
  emit({ online, counts: await queueCounts() });
  if (!online) return;
  emit({ syncing: true });
  const db = await getDb();
  try {
    const rows = await db.getAllAsync("SELECT * FROM outbox WHERE state = 'pending' ORDER BY id");
    for (const row of rows) {
      try {
        if (row.kind === 'duty') {
          await sendDuty(row.body);
          await db.runAsync("UPDATE outbox SET state = 'sent', sent_at = ? WHERE id = ?", Date.now(), row.id);
        } else {
          const res = await sendViolation(row);
          await db.runAsync(
            "UPDATE outbox SET state = 'sent', sent_at = ?, case_no = ?, last_error = NULL WHERE id = ?",
            Date.now(), res.case_no ?? null, row.id,
          );
          // Server has the photos (and checked their hashes): free the phone's storage.
          if (row.folder) await RsafeDetector.deleteUploadedEvidence(row.folder).catch(() => {});
        }
        emit({ blocked: null, counts: await queueCounts() });
      } catch (e) {
        const code = e instanceof ApiError ? e.code : 'network';
        await db.runAsync('UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE id = ?', String(e.message), row.id);
        if (PERMANENT.has(code)) {
          await db.runAsync("UPDATE outbox SET state = 'rejected' WHERE id = ?", row.id);
          continue;
        }
        if (DEVICE_BLOCKED.has(code)) emit({ blocked: code });
        break; // keep order: duty sessions before the violations that use them
      }
    }
    const last = await kvGet('last_seq');
    const counts = await queueCounts();
    await sendHeartbeat({ last_seq: Number(last || 0), pending_count: counts.pending, app_version: APP_VERSION })
      .catch(() => {});
    await kvSet('last_sync', Date.now());
    emit({ counts, lastSyncAt: Date.now() });
  } finally {
    emit({ syncing: false });
  }
}

export function startSync() {
  if (started) return;
  started = true;
  Network.addNetworkStateListener(() => { syncNow(); });
  setInterval(() => { syncNow(); }, 30000);
  syncNow();
}
