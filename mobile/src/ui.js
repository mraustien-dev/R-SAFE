// Shared look for every screen.
// Palette, 60/30/10: night black base, warm white text, signal orange accent (used sparingly:
// the main action, the selected item, links).
import { useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export const colors = {
  bg: '#0A0909',        // night black, 60%
  card: '#151313',      // raised surface, a step above the base
  ink: '#F3EFEA',       // warm white, 30%
  muted: '#9A938C',     // warm white at reduced strength, for secondary text
  line: '#2A2625',      // hairline borders
  primary: '#FF5A1F',   // signal orange, 10%
  primarySoft: 'rgba(255, 90, 31, 0.10)',
  onPrimary: '#0A0909', // text on orange: black reads better than white
  danger: '#F2564B',
  good: '#4CC38A',
  warn: '#F2B84B',
};

export function Screen({ children, style }) {
  return <SafeAreaView style={[styles.screen, style]}>{children}</SafeAreaView>;
}

/** Scrolling form column: fits short landscape screens, and stays readable on wide ones. */
export function Form({ children }) {
  return (
    <ScrollView
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={{ flexGrow: 1, width: '100%', maxWidth: 560, alignSelf: 'center' }}
    >
      {children}
    </ScrollView>
  );
}

export function Title({ children, sub }) {
  return (
    <View style={{ marginBottom: 28, marginTop: 8 }}>
      <View style={styles.titleMark} />
      <Text style={styles.title}>{children}</Text>
      {sub ? <Text style={styles.sub}>{sub}</Text> : null}
    </View>
  );
}

/** kind: 'primary' (orange), 'secondary' (outline), 'danger' (red outline). */
export function Button({ title, onPress, busy, kind = 'primary', disabled }) {
  const primary = kind === 'primary';
  const fg = primary ? colors.onPrimary : kind === 'danger' ? colors.danger : colors.ink;
  return (
    <Pressable
      onPress={onPress}
      disabled={busy || disabled}
      accessibilityRole="button"
      style={({ pressed }) => [
        styles.button,
        primary
          ? { backgroundColor: colors.primary }
          : { backgroundColor: 'transparent', borderWidth: 1, borderColor: kind === 'danger' ? colors.danger : colors.line },
        { opacity: disabled ? 0.4 : pressed ? 0.8 : 1 },
      ]}
    >
      {busy ? <ActivityIndicator color={fg} /> : <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>}
    </Pressable>
  );
}

export function Field({ label, onFocus, onBlur, ...props }) {
  const [focused, setFocused] = useState(false);
  return (
    <View style={{ marginBottom: 16 }}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        placeholderTextColor={colors.muted}
        selectionColor={colors.primary}
        cursorColor={colors.primary}
        style={[styles.input, focused && { borderColor: colors.primary }]}
        onFocus={(e) => { setFocused(true); onFocus?.(e); }}
        onBlur={(e) => { setFocused(false); onBlur?.(e); }}
        {...props}
      />
    </View>
  );
}

export function ErrorText({ children }) {
  return children ? <Text style={styles.error}>{children}</Text> : null;
}

export const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg, padding: 20 },
  titleMark: { width: 28, height: 3, borderRadius: 2, backgroundColor: colors.primary, marginBottom: 14 },
  title: { fontSize: 28, fontWeight: '700', color: colors.ink, letterSpacing: -0.4 },
  sub: { fontSize: 15, color: colors.muted, marginTop: 8, lineHeight: 22 },
  label: { fontSize: 13, fontWeight: '600', color: colors.muted, marginBottom: 8, letterSpacing: 0.3 },
  input: {
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line, borderRadius: 12,
    paddingHorizontal: 14, paddingVertical: 13, fontSize: 16, color: colors.ink,
  },
  button: { borderRadius: 12, paddingVertical: 14, alignItems: 'center', marginTop: 10, minHeight: 52, justifyContent: 'center' },
  buttonText: { fontSize: 16, fontWeight: '700', letterSpacing: 0.2 },
  error: { color: colors.danger, marginTop: 12, fontSize: 14, lineHeight: 20 },
  card: { backgroundColor: colors.card, borderRadius: 14, padding: 16, borderWidth: 1, borderColor: colors.line },
});
