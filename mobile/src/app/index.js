// Decides where to go: register the phone -> sign in -> duty.
import { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Text, View } from 'react-native';
import { Redirect } from 'expo-router';
import { ensureIdentity } from '../services/device';
import { currentSession } from '../services/session';
import { Screen, colors } from '../ui';

export default function Index() {
  const [target, setTarget] = useState(null);

  useEffect(() => {
    ensureIdentity().then((id) => {
      if (!id.camera) setTarget('/register');
      else if (!currentSession().officer) setTarget('/login');
      else setTarget('/duty');
    });
  }, []);

  if (target) return <Redirect href={target} />;
  // Opening screen: the logo sits exactly where the splash screen drew it (centered, 200 wide),
  // so the hand-off from splash to app is seamless. Name and spinner sit below it.
  return (
    <Screen style={{ alignItems: 'center', justifyContent: 'center' }}>
      <Image source={require('../../assets/splash-icon.png')} style={{ width: 200, height: 200 }} resizeMode="contain" />
      <View style={{ position: 'absolute', top: '50%', marginTop: 124, alignItems: 'center' }}>
        <Text style={{ color: colors.ink, fontSize: 22, fontWeight: '700', letterSpacing: 4 }}>R-SAFE</Text>
        <Text style={{ color: colors.muted, fontSize: 13, marginTop: 6, letterSpacing: 0.3 }}>
          Helmet violation capture
        </Text>
      </View>
      <View style={{ position: 'absolute', bottom: 56, alignItems: 'center' }}>
        <ActivityIndicator size="small" color={colors.primary} />
        <Text style={{ color: colors.muted, fontSize: 12, marginTop: 10 }}>Checking this phone…</Text>
      </View>
    </Screen>
  );
}
