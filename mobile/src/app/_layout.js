import { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { startSync, syncNow } from '../services/sync';
import { recoverEvidence } from '../services/session';

export default function RootLayout() {
  useEffect(() => {
    startSync(); // uploads run for as long as the app is open, on every screen
    recoverEvidence()
      .then((n) => { if (n) syncNow(); })
      .catch((e) => console.warn('[R-SAFE] evidence recovery failed', e));
  }, []);
  return (
    <>
      <StatusBar style="dark" />
      <Stack screenOptions={{ headerShown: false, animation: 'fade' }} />
    </>
  );
}
