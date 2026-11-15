// Calls to the device-api Edge Function. Every call is signed with the phone's key.
import { API_URL, APP_VERSION, ensureIdentity, extra, signText } from './device';
import { requestMessage } from '../lib/record';
import { kvSet } from './db';
import { File } from 'expo-file-system';

export class ApiError extends Error {
  constructor(status, code, detail) {
    super(detail ? `${code}: ${detail}` : code);
    this.status = status;
    this.code = code;
  }
}

async function handle(res) {
  let body = {};
  try { body = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new ApiError(res.status, body.error ?? `http_${res.status}`, body.detail);
  await kvSet('last_contact', Date.now()); // server reached: offline-sign-in clock restarts
  return body;
}

async function signedJson(route, payload, extraHeaders = {}) {
  const { deviceId } = await ensureIdentity();
  const body = JSON.stringify(payload);
  const ts = String(Date.now());
  const res = await fetch(`${API_URL}/${route}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-device-id': deviceId,
      'x-timestamp': ts,
      'x-signature': await signText(requestMessage(route, ts, body)),
      ...extraHeaders,
    },
    body,
  });
  return handle(res);
}

/** One-time: link this phone to the camera row the admin made the code for. */
export async function registerDevice(code) {
  const { deviceId, publicKey } = await ensureIdentity();
  const body = JSON.stringify({ code: code.trim().toUpperCase(), device_id: deviceId, public_key: publicKey, app_version: APP_VERSION });
  const res = await fetch(`${API_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-signature': await signText(`register\n${body}`) },
    body,
  });
  return handle(res);
}

/** Officer email + password -> Supabase access token (online only). */
export async function officerPasswordSignIn(email, password) {
  const res = await fetch(`${extra.supabaseUrl}/auth/v1/token?grant_type=password`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', apikey: extra.supabasePublishableKey },
    body: JSON.stringify({ email: email.trim(), password }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, body.error_description || body.msg || 'sign_in_failed');
  return body.access_token;
}

export const enrollOfficer = (accessToken) =>
  signedJson('enroll', { at: Date.now() }, { authorization: `Bearer ${accessToken}` });

export const sendDuty = (bodyText) => signedJsonRaw('duty', bodyText);

export const sendHeartbeat = (payload) => signedJson('heartbeat', payload);

// duty rows are stored as the exact JSON text that is sent
async function signedJsonRaw(route, body) {
  const { deviceId } = await ensureIdentity();
  const ts = String(Date.now());
  const res = await fetch(`${API_URL}/${route}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-device-id': deviceId,
      'x-timestamp': ts,
      'x-signature': await signText(requestMessage(route, ts, body)),
    },
    body,
  });
  return handle(res);
}

/** Violation: the record signed at capture + the photos. */
export async function sendViolation(row) {
  const form = new FormData();
  form.append('record', row.body);
  form.append('record_signature', row.signature);
  const files = JSON.parse(row.files || '{}');
  for (const [name, path] of Object.entries(files)) {
    // Expo's fetch (the global fetch in this SDK) rejects RN-style { uri } parts; File implements Blob.
    form.append(name, new File(path.startsWith('file://') ? path : `file://${path}`));
  }
  const res = await fetch(`${API_URL}/violation`, { method: 'POST', body: form });
  return handle(res);
}
