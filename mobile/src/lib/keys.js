// Phone identity key (ECDSA P-256) and officer PIN hashing.
// Pure JS (no React Native imports) so it is unit-tested with Node.
import { p256 } from '@noble/curves/p256';
import { sha256 } from '@noble/hashes/sha256';
import { pbkdf2 } from '@noble/hashes/pbkdf2';
import { bytesToHex } from '@noble/hashes/utils';

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

export function toBase64(bytes) {
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
    out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63];
    out += i + 1 < bytes.length ? B64[(n >> 6) & 63] : '=';
    out += i + 2 < bytes.length ? B64[n & 63] : '=';
  }
  return out;
}

export function fromBase64(text) {
  const clean = text.replace(/=+$/, '');
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let bits = 0;
  let value = 0;
  let o = 0;
  for (const ch of clean) {
    value = (value << 6) | B64.indexOf(ch);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (value >> bits) & 0xff;
    }
  }
  return out;
}

const utf8 = (s) => new TextEncoder().encode(s);

/** randomBytes32: 32 secure random bytes (expo-crypto on the phone). */
export function createKeyPair(randomBytes32) {
  const priv = randomBytes32();
  if (!p256.utils.isValidPrivateKey(priv)) return createKeyPair(randomBytes32);
  return {
    privateKey: toBase64(priv),
    publicKey: toBase64(p256.getPublicKey(priv, false)), // 65-byte uncompressed point
  };
}

/** Base64 raw r||s signature over SHA-256(message), as the server's WebCrypto expects. */
export function sign(privateKeyB64, message) {
  const sig = p256.sign(sha256(utf8(message)), fromBase64(privateKeyB64), { prehash: false });
  return toBase64(sig.toCompactRawBytes());
}

export function verify(publicKeyB64, signatureB64, message) {
  return p256.verify(fromBase64(signatureB64), sha256(utf8(message)), fromBase64(publicKeyB64), {
    prehash: false,
  });
}

export const sha256Hex = (text) => bytesToHex(sha256(utf8(text)));

// PIN: never stored, only a slow salted hash (PBKDF2-SHA256).
export const PIN_ITERATIONS = 20000;

export function hashPin(pin, saltBytes, iterations = PIN_ITERATIONS) {
  return bytesToHex(pbkdf2(sha256, utf8(pin), saltBytes, { c: iterations, dkLen: 32 }));
}

export function pinMatches(pin, saltHex, hashHex, iterations = PIN_ITERATIONS) {
  const salt = Uint8Array.from(saltHex.match(/../g).map((h) => parseInt(h, 16)));
  const got = hashPin(pin, salt, iterations);
  let diff = got.length ^ hashHex.length;
  for (let i = 0; i < Math.min(got.length, hashHex.length); i++) {
    diff |= got.charCodeAt(i) ^ hashHex.charCodeAt(i);
  }
  return diff === 0;
}

export const isValidPin = (pin) => /^\d{4,6}$/.test(pin);
