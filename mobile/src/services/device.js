// This phone's identity: device id + P-256 key, kept in Android Keystore-backed SecureStore.
import * as SecureStore from 'expo-secure-store';
import * as Crypto from 'expo-crypto';
import Constants from 'expo-constants';
import { createKeyPair, sign } from '../lib/keys';

const KEYS = {
  deviceId: 'rsafe_device_id',
  privateKey: 'rsafe_device_private_key',
  publicKey: 'rsafe_device_public_key',
  camera: 'rsafe_device_camera',
};

const options = { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY };

export const extra = Constants.expoConfig?.extra ?? {};
export const APP_VERSION = Constants.expoConfig?.version ?? '0.0.0';
export const API_URL = `${extra.supabaseUrl}/functions/v1/device-api`;

let cache = null;

/** Creates the key pair the first time; the private key never leaves the phone. */
export async function ensureIdentity() {
  if (cache) return cache;
  let deviceId = await SecureStore.getItemAsync(KEYS.deviceId);
  let privateKey = await SecureStore.getItemAsync(KEYS.privateKey);
  let publicKey = await SecureStore.getItemAsync(KEYS.publicKey);
  if (!deviceId || !privateKey || !publicKey) {
    deviceId = Crypto.randomUUID();
    ({ privateKey, publicKey } = createKeyPair(() => Crypto.getRandomBytes(32)));
    await SecureStore.setItemAsync(KEYS.deviceId, deviceId, options);
    await SecureStore.setItemAsync(KEYS.privateKey, privateKey, options);
    await SecureStore.setItemAsync(KEYS.publicKey, publicKey, options);
  }
  const cameraText = await SecureStore.getItemAsync(KEYS.camera);
  cache = { deviceId, privateKey, publicKey, camera: cameraText ? JSON.parse(cameraText) : null };
  return cache;
}

export async function saveCamera(camera) {
  await SecureStore.setItemAsync(KEYS.camera, JSON.stringify(camera), options);
  if (cache) cache.camera = camera;
}

export async function signText(text) {
  const { privateKey } = await ensureIdentity();
  return sign(privateKey, text);
}
