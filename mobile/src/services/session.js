// Officers on this phone (Option A: sign in online once, then PIN offline), duty sessions,
// and turning a detection into a signed, queued record.
import * as Crypto from 'expo-crypto';
import * as Location from 'expo-location';
import { getDb, kvGet, kvSet, nextSeq } from './db';
import { enrollOfficer, officerPasswordSignIn } from './api';
import { APP_VERSION, ensureIdentity, signText } from './device';
import { hashPin, isValidPin, pinMatches } from '../lib/keys';
import { buildRecord, MAX_PIN_ATTEMPTS, OFFLINE_LIMIT_MS } from '../lib/record';
import { syncNow } from './sync';
import RsafeDetector from '../../modules/rsafe-detector';

const listeners = new Set();
let state = { officer: null, duty: null };

export function onSession(fn) {
  listeners.add(fn);
  fn(state);
  return () => listeners.delete(fn);
}
const emit = (patch) => {
  state = { ...state, ...patch };
  listeners.forEach((fn) => fn(state));
};
export const currentSession = () => state;

// ---------- officers ------------------------------------------------------------------

export async function listOfficers() {
  const db = await getDb();
  return db.getAllAsync('SELECT id, email, full_name, failed FROM officers ORDER BY full_name');
}

/** Online: check email/password with Supabase, link officer to this phone, store a PIN hash. */
export async function enrollWithPassword(email, password, pin) {
  if (!isValidPin(pin)) throw new Error('PIN must be 4 to 6 digits');
  const token = await officerPasswordSignIn(email, password);
  const officer = await enrollOfficer(token); // server checks the officer role
  const salt = Crypto.getRandomBytes(16);
  const saltHex = Array.from(salt, (b) => b.toString(16).padStart(2, '0')).join('');
  const db = await getDb();
  await db.runAsync(
    `INSERT OR REPLACE INTO officers (id, email, full_name, pin_salt, pin_hash, failed, enrolled_at)
     VALUES (?, ?, ?, ?, ?, 0, ?)`,
    officer.officer_id, officer.email ?? email, officer.full_name ?? email, saltHex, hashPin(pin, salt), Date.now(),
  );
  return officer;
}

/** Hours left before this phone must reach the server again (offline sign-in limit). */
export async function offlineHoursLeft() {
  const last = Number((await kvGet('last_contact')) || 0);
  return Math.max(0, (last + OFFLINE_LIMIT_MS - Date.now()) / 3600000);
}

/** Offline-capable: check the PIN on the phone. */
export async function signInWithPin(officerId, pin) {
  const db = await getDb();
  const o = await db.getFirstAsync('SELECT * FROM officers WHERE id = ?', officerId);
  if (!o) throw new Error('Officer not on this phone');
  if (o.failed >= MAX_PIN_ATTEMPTS) {
    throw new Error('Too many wrong PINs. Sign in online with your password again.');
  }
  if ((await offlineHoursLeft()) <= 0) {
    throw new Error('This phone has been offline for over 3 days. Connect to the internet once to continue.');
  }
  if (!pinMatches(pin, o.pin_salt, o.pin_hash)) {
    await db.runAsync('UPDATE officers SET failed = failed + 1 WHERE id = ?', officerId);
    const left = MAX_PIN_ATTEMPTS - o.failed - 1;
    throw new Error(left > 0 ? `Wrong PIN. ${left} tries left.` : 'Wrong PIN. This officer must sign in online again.');
  }
  await db.runAsync('UPDATE officers SET failed = 0 WHERE id = ?', officerId);
  const officer = { id: o.id, email: o.email, full_name: o.full_name };
  emit({ officer });
  // A duty left open (app closed) by this officer continues; one by someone else is closed.
  const open = await db.getFirstAsync('SELECT * FROM duty WHERE ended_at IS NULL ORDER BY started_at DESC');
  if (open && open.officer_id !== officer.id) await endDuty(open.id);
  emit({ duty: open && open.officer_id === officer.id ? open : null });
  return officer;
}

export async function signOut() {
  if (state.duty) await endDuty(state.duty.id);
  emit({ officer: null, duty: null });
}

// ---------- duty ----------------------------------------------------------------------

async function queueDuty(duty) {
  const db = await getDb();
  const body = JSON.stringify({
    id: duty.id,
    officer_id: duty.officer_id,
    started_at: new Date(duty.started_at).toISOString(),
    ended_at: duty.ended_at ? new Date(duty.ended_at).toISOString() : null,
  });
  await db.runAsync("INSERT INTO outbox (kind, body, created_at) VALUES ('duty', ?, ?)", body, Date.now());
}

export async function startDuty() {
  if (!state.officer) throw new Error('Not signed in');
  const duty = { id: Crypto.randomUUID(), officer_id: state.officer.id, started_at: Date.now(), ended_at: null };
  const db = await getDb();
  await db.runAsync('INSERT INTO duty (id, officer_id, started_at) VALUES (?, ?, ?)', duty.id, duty.officer_id, duty.started_at);
  await queueDuty(duty);
  emit({ duty });
  syncNow();
  return duty;
}

export async function endDuty(id = state.duty?.id) {
  if (!id) return;
  const db = await getDb();
  const ended = Date.now();
  await db.runAsync('UPDATE duty SET ended_at = ? WHERE id = ? AND ended_at IS NULL', ended, id);
  const duty = await db.getFirstAsync('SELECT * FROM duty WHERE id = ?', id);
  if (duty) await queueDuty(duty);
  if (state.duty?.id === id) emit({ duty: null });
  syncNow();
}

// ---------- detections ----------------------------------------------------------------

let lastLocation = null;
export async function refreshLocation() {
  try {
    const { granted } = await Location.requestForegroundPermissionsAsync();
    if (!granted) return;
    const pos = (await Location.getLastKnownPositionAsync()) ?? (await Location.getCurrentPositionAsync({}));
    if (pos) {
      lastLocation = {
        lat: pos.coords.latitude,
        lng: pos.coords.longitude,
        accuracy_m: pos.coords.accuracy,
        at: new Date(pos.timestamp).toISOString(),
      };
    }
  } catch { /* no GPS: record keeps location null */ }
}

/**
 * Called for every finished violation from the native detector. The record is signed
 * immediately, so nothing about it (including photo hashes) can change before upload.
 */
export async function recordViolation(event, ctx = {}) {
  const { deviceId } = await ensureIdentity();
  const db = await getDb();
  // already queued (e.g. recovered twice)? never create a second record for one folder
  if (await db.getFirstAsync('SELECT id FROM outbox WHERE folder = ?', event.folder)) return null;
  const seq = await nextSeq();
  const { text, files } = buildRecord(event, {
    deviceId,
    seq,
    officerId: ctx.officerId !== undefined ? ctx.officerId : state.officer?.id ?? null,
    dutySessionId: ctx.dutyId !== undefined ? ctx.dutyId : state.duty?.id ?? null,
    appVersion: APP_VERSION,
    location: lastLocation,
  });
  const signature = await signText(text);
  await db.runAsync(
    `INSERT INTO outbox (kind, seq, body, signature, files, folder, created_at)
     VALUES ('violation', ?, ?, ?, ?, ?, ?)`,
    seq, text, signature, JSON.stringify(files), event.folder, Date.now(),
  );
  await kvSet('last_violation_at', Date.now());
  syncNow(); // live upload when there is internet
  return seq;
}

/**
 * Evidence saved by the detector but never queued (app closed or crashed right after a
 * detection) is queued now, credited to the duty session that was open at that moment.
 */
export async function recoverEvidence() {
  const db = await getDb();
  const texts = await RsafeDetector.listEvidence();
  let recovered = 0;
  for (const text of texts) {
    let event;
    try { event = JSON.parse(text); } catch { continue; }
    if (await db.getFirstAsync('SELECT id FROM outbox WHERE folder = ?', event.folder)) continue;
    const duty = await db.getFirstAsync(
      'SELECT * FROM duty WHERE started_at <= ? AND (ended_at IS NULL OR ended_at >= ?) ORDER BY started_at DESC',
      event.detectedAtMs, event.detectedAtMs,
    );
    if (await recordViolation(event, { officerId: duty?.officer_id ?? null, dutyId: duty?.id ?? null })) recovered++;
  }
  return recovered;
}
