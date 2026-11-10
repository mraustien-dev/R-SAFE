import test from 'node:test';
import assert from 'node:assert/strict';
import { randomBytes, webcrypto } from 'node:crypto';
import {
  createKeyPair, sign, verify, toBase64, fromBase64, hashPin, pinMatches, isValidPin,
} from '../src/lib/keys.js';
import { phTime, violationId, buildRecord, requestMessage } from '../src/lib/record.js';

const rnd = () => new Uint8Array(randomBytes(32));

// Exactly what backend/supabase/functions/device-api does.
async function serverVerify(pub, sig, message) {
  const key = await webcrypto.subtle.importKey(
    'raw', fromBase64(pub), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify'],
  );
  return webcrypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, fromBase64(sig), new TextEncoder().encode(message));
}

test('base64 round trip matches Node', () => {
  for (const n of [0, 1, 2, 3, 31, 64, 65]) {
    const b = new Uint8Array(randomBytes(n));
    assert.equal(toBase64(b), Buffer.from(b).toString('base64'));
    assert.deepEqual(fromBase64(toBase64(b)), b);
  }
});

test('phone signature passes the server check and fails when changed', async () => {
  const { privateKey, publicKey } = createKeyPair(rnd);
  const msg = requestMessage('duty', 1700000000000, '{"id":"x"}');
  const sig = sign(privateKey, msg);
  assert.ok(verify(publicKey, sig, msg));
  assert.ok(await serverVerify(publicKey, sig, msg));
  assert.equal(await serverVerify(publicKey, sig, msg + ' '), false);
});

test('PIN hash', () => {
  const salt = new Uint8Array(randomBytes(16));
  const saltHex = Buffer.from(salt).toString('hex');
  const hash = hashPin('1234', salt, 1000);
  assert.ok(pinMatches('1234', saltHex, hash, 1000));
  assert.equal(pinMatches('1235', saltHex, hash, 1000), false);
  assert.ok(isValidPin('123456'));
  assert.equal(isValidPin('12a4'), false);
  assert.equal(isValidPin('123'), false);
});

test('Philippine time and violation id', () => {
  const ms = Date.UTC(2026, 10, 6, 6, 32, 12); // 14:32:12 in UTC+8
  assert.equal(phTime(ms), '2026-11-06T14:32:12+08:00');
  assert.equal(violationId(ms, 7), 'V-20261106-143212-0007');
});

test('record keeps hashes, not local paths', () => {
  const event = {
    trackId: 3,
    detectedAtMs: Date.UTC(2026, 10, 6, 6, 0, 0),
    riders: [{ position: 1, helmet: 'no_helmet', confidence: 0.8, frames_observed: 4 }],
    motorcycle: { confidence: 0.9, bbox: [1, 2, 3, 4] },
    plate: { status: 'read', text: 'ABC123', ocr_confidence: 0.9 },
    images: { frame: { path: '/data/x/frame.jpg', sha256: 'aa', bytes: 10 } },
    models: { vehicle: 'v' },
  };
  const { text, files } = buildRecord(event, {
    deviceId: 'dev', seq: 5, officerId: 'off', dutySessionId: 'duty', appVersion: '1.0.0', location: null,
  });
  const rec = JSON.parse(text);
  assert.equal(rec.seq, 5);
  assert.equal(rec.device_id, 'dev');
  assert.deepEqual(rec.images.frame, { sha256: 'aa', bytes: 10 });
  assert.equal(text.includes('/data/x'), false);
  assert.equal(files.frame, '/data/x/frame.jpg');
});
