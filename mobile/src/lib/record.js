// Builds the violation record the phone signs at capture time.
// Pure JS (no React Native imports) so it is unit-tested with Node.

const PH_OFFSET_MS = 8 * 60 * 60 * 1000;
const pad = (n, w = 2) => String(n).padStart(w, '0');

/** "2026-11-06T14:32:12+08:00" (Philippine time, like the laptop engine). */
export function phTime(ms) {
  const d = new Date(ms + PH_OFFSET_MS);
  return (
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}` +
    `T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}+08:00`
  );
}

/** "V-20261106-143212-0007" (same pattern as ai-engine/docs/violation_schema.md). */
export function violationId(ms, trackId) {
  const t = phTime(ms);
  return `V-${t.slice(0, 10).replace(/-/g, '')}-${t.slice(11, 19).replace(/:/g, '')}-${pad(trackId % 10000, 4)}`;
}

/**
 * event: what the native detector reported (onViolation).
 * Returns { text, files } — `text` is the exact JSON string to sign and upload;
 * `files` maps image name -> local path (not part of the signed record).
 */
export function buildRecord(event, { deviceId, seq, officerId, dutySessionId, appVersion, location }) {
  const images = {};
  const files = {};
  for (const [name, img] of Object.entries(event.images ?? {})) {
    images[name] = { sha256: img.sha256, bytes: img.bytes };
    files[name] = img.path;
  }
  const record = {
    schema: 'rsafe-mobile-1',
    device_id: deviceId,
    seq,
    violation_id: violationId(event.detectedAtMs, event.trackId),
    type: 'no_helmet',
    status: 'pending_review',
    detected_at: phTime(event.detectedAtMs),
    officer_id: officerId ?? null,
    duty_session_id: dutySessionId ?? null,
    track_id: event.trackId,
    riders: event.riders,
    motorcycle: event.motorcycle,
    plate: event.plate,
    images,
    models: event.models,
    app_version: appVersion,
    location: location ?? null,
  };
  return { text: JSON.stringify(record), files };
}

/** Request signature input for /enroll, /duty, /heartbeat (matches device-api). */
export const requestMessage = (route, timestamp, body) => `${route}\n${timestamp}\n${body}`;

export const OFFLINE_LIMIT_MS = 72 * 60 * 60 * 1000;
export const MAX_PIN_ATTEMPTS = 5;
