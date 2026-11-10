// JS side of the native R-SAFE detector (Kotlin, android/).
import { requireNativeModule, requireNativeViewManager } from 'expo-modules-core';

// <RsafeCameraView active onStatus onViolation onError style />
export const RsafeCameraView = requireNativeViewManager('RsafeDetector');

const RsafeDetector = requireNativeModule('RsafeDetector');
export default RsafeDetector;
