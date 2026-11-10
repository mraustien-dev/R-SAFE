// R-SAFE device API — the ONLY way phones write to the database.
//
// Routes (POST, JSON unless noted):
//   /device-api/register   one-time phone registration with an admin's code
//   /device-api/enroll     officer signed in online on this phone (officer JWT + device signature)
//   /device-api/duty       duty session start/end (device signature)
//   /device-api/violation  multipart: record + record_signature + frame/rider/plate jpg
//   /device-api/heartbeat  phone status: newest seq, queue size (device signature)
//
// Every request after /register is signed with the phone's P-256 key (ECDSA, SHA-256,
// raw r||s signature, base64). The server never trusts the phone's word alone:
// it checks the signature, image hashes, officer enrolment and role, and flags
// (never silently drops) anything that does not add up.

import { createClient, SupabaseClient } from "npm:@supabase/supabase-js@2.49.4";

const BUCKET = "case-evidence";
const IMAGE_FIELDS = ["frame", "rider", "plate"] as const;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

const admin: SupabaseClient = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  { auth: { persistSession: false } },
);

// ---------- helpers -------------------------------------------------------------

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

const enc = new TextEncoder();

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function sha256Hex(data: Uint8Array | string): Promise<string> {
  const bytes = typeof data === "string" ? enc.encode(data) : data;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function verifySignature(
  publicKeyB64: string,
  signatureB64: string,
  message: string,
): Promise<boolean> {
  try {
    const key = await crypto.subtle.importKey(
      "raw",
      b64ToBytes(publicKeyB64),
      { name: "ECDSA", namedCurve: "P-256" },
      false,
      ["verify"],
    );
    return await crypto.subtle.verify(
      { name: "ECDSA", hash: "SHA-256" },
      key,
      b64ToBytes(signatureB64),
      enc.encode(message),
    );
  } catch {
    return false;
  }
}

type Camera = {
  id: string;
  name: string;
  location_name: string | null;
  is_active: boolean | null;
  device_public_key: string;
};

async function cameraByDevice(deviceId: string): Promise<Camera | null> {
  const { data } = await admin
    .from("cameras")
    .select("id, name, location_name, is_active, device_public_key")
    .eq("device_id", deviceId)
    .not("device_public_key", "is", null)
    .maybeSingle();
  return data as Camera | null;
}

// Signed JSON request: headers x-device-id, x-timestamp, x-signature over
// `${route}\n${timestamp}\n${rawBody}`.
async function authDevice(
  req: Request,
  route: string,
): Promise<{ camera: Camera; body: any } | Response> {
  const deviceId = req.headers.get("x-device-id") ?? "";
  const ts = req.headers.get("x-timestamp") ?? "";
  const sig = req.headers.get("x-signature") ?? "";
  const raw = await req.text();
  const camera = await cameraByDevice(deviceId);
  if (!camera) return json(401, { error: "unknown_device" });
  if (!(await verifySignature(camera.device_public_key, sig, `${route}\n${ts}\n${raw}`))) {
    return json(401, { error: "bad_signature" });
  }
  if (camera.is_active === false) return json(403, { error: "device_disabled" });
  let body: any;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, { error: "bad_json" });
  }
  return { camera, body };
}

async function hasRole(userId: string, role: "admin" | "officer"): Promise<boolean> {
  const { data } = await admin
    .from("user_roles")
    .select("id")
    .eq("user_id", userId)
    .eq("role", role)
    .maybeSingle();
  return !!data;
}

// ---------- routes --------------------------------------------------------------

async function register(req: Request): Promise<Response> {
  const raw = await req.text();
  let body: any;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, { error: "bad_json" });
  }
  const { code, device_id, public_key, app_version } = body ?? {};
  const sig = req.headers.get("x-signature") ?? "";
  if (!code || !device_id || !public_key) return json(400, { error: "missing_fields" });
  // proves the phone holds the private key it is registering
  if (!(await verifySignature(public_key, sig, `register\n${raw}`))) {
    return json(401, { error: "bad_signature" });
  }
  const codeHash = await sha256Hex(String(code).trim().toUpperCase());
  const { data: cam } = await admin
    .from("cameras")
    .select("id, name, location_name, provision_expires_at")
    .eq("provision_code_hash", codeHash)
    .maybeSingle();
  if (!cam || !cam.provision_expires_at || new Date(cam.provision_expires_at) < new Date()) {
    return json(403, { error: "invalid_or_expired_code" });
  }
  const { error } = await admin
    .from("cameras")
    .update({
      device_id,
      device_public_key: public_key,
      provisioned_at: new Date().toISOString(),
      provision_code_hash: null,
      provision_expires_at: null,
      app_version: app_version ?? null,
      last_seq: 0,
      pending_count: 0,
      last_seen_at: new Date().toISOString(),
    })
    .eq("id", cam.id);
  if (error) return json(409, { error: "device_id_in_use" });
  return json(200, { camera_id: cam.id, name: cam.name, location_name: cam.location_name });
}

async function enroll(req: Request): Promise<Response> {
  const auth = await authDevice(req, "enroll");
  if (auth instanceof Response) return auth;
  const token = (req.headers.get("authorization") ?? "").replace(/^Bearer\s+/i, "");
  const { data: userData, error } = await admin.auth.getUser(token);
  if (error || !userData?.user) return json(401, { error: "bad_officer_token" });
  const user = userData.user;
  if (!(await hasRole(user.id, "officer")) && !(await hasRole(user.id, "admin"))) {
    return json(403, { error: "not_an_officer" });
  }
  const { data: existing } = await admin
    .from("device_officers")
    .select("revoked_at")
    .eq("camera_id", auth.camera.id)
    .eq("officer_id", user.id)
    .maybeSingle();
  if (existing?.revoked_at) return json(403, { error: "officer_revoked_on_device" });
  if (!existing) {
    await admin.from("device_officers").insert({ camera_id: auth.camera.id, officer_id: user.id });
  }
  const { data: profile } = await admin
    .from("users")
    .select("full_name")
    .eq("id", user.id)
    .maybeSingle();
  return json(200, {
    officer_id: user.id,
    email: user.email,
    full_name: profile?.full_name ?? user.email,
  });
}

async function duty(req: Request): Promise<Response> {
  const auth = await authDevice(req, "duty");
  if (auth instanceof Response) return auth;
  const { id, officer_id, started_at, ended_at } = auth.body ?? {};
  if (!id || !officer_id || !started_at) return json(400, { error: "missing_fields" });
  const { error } = await admin.from("duty_sessions").upsert(
    {
      id,
      camera_id: auth.camera.id,
      officer_id,
      started_at,
      ended_at: ended_at ?? null,
    },
    { onConflict: "id" },
  );
  if (error) return json(422, { error: "duty_rejected", detail: error.message });
  return json(200, { ok: true });
}

async function heartbeat(req: Request): Promise<Response> {
  const auth = await authDevice(req, "heartbeat");
  if (auth instanceof Response) return auth;
  const { last_seq, pending_count, app_version } = auth.body ?? {};
  await admin
    .from("cameras")
    .update({
      last_seq: Number(last_seq) || 0,
      pending_count: Number(pending_count) || 0,
      app_version: app_version ?? null,
      last_seen_at: new Date().toISOString(),
    })
    .eq("id", auth.camera.id);
  return json(200, { ok: true, server_time: new Date().toISOString() });
}

async function violation(req: Request): Promise<Response> {
  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return json(400, { error: "bad_multipart" });
  }
  const recordText = String(form.get("record") ?? "");
  const signature = String(form.get("record_signature") ?? "");
  let record: any;
  try {
    record = JSON.parse(recordText);
  } catch {
    return json(400, { error: "bad_record" });
  }

  const camera = await cameraByDevice(String(record.device_id ?? ""));
  if (!camera) return json(401, { error: "unknown_device" });
  // The record was signed on the phone at capture time.
  if (!(await verifySignature(camera.device_public_key, signature, recordText))) {
    return json(401, { error: "bad_signature" });
  }
  if (camera.is_active === false) return json(403, { error: "device_disabled" });

  const seq = Number(record.seq);
  if (!Number.isInteger(seq) || seq < 1) return json(400, { error: "bad_seq" });

  // Idempotent: the phone may resend after a dropped connection.
  const { data: dup } = await admin
    .from("cases")
    .select("case_no")
    .eq("camera_id", camera.id)
    .eq("device_seq", seq)
    .maybeSingle();
  if (dup) return json(200, { case_no: dup.case_no, duplicate: true });

  const notes: string[] = [];
  const caseNo = `RS-${camera.id.slice(0, 4).toUpperCase()}-${String(seq).padStart(6, "0")}`;

  // Images: hash what arrived and compare with the hashes signed at capture.
  const uploads: { name: string; path: string; bytes: Uint8Array; sha: string }[] = [];
  for (const name of IMAGE_FIELDS) {
    const expected = record.images?.[name];
    const file = form.get(name);
    if (!expected) continue;
    if (!(file instanceof File)) {
      notes.push(`missing_image:${name}`);
      continue;
    }
    if (file.size > MAX_IMAGE_BYTES) return json(413, { error: `image_too_large:${name}` });
    const bytes = new Uint8Array(await file.arrayBuffer());
    const sha = await sha256Hex(bytes);
    if (sha !== expected.sha256) notes.push(`image_hash_mismatch:${name}`);
    uploads.push({ name, path: `${camera.id}/${caseNo}/${name}.jpg`, bytes, sha });
  }

  // Officer: must have enrolled online on THIS phone and still hold a staff role.
  const officerId: string | null = record.officer_id ?? null;
  if (!officerId) {
    notes.push("no_officer");
  } else {
    const { data: enrol } = await admin
      .from("device_officers")
      .select("revoked_at")
      .eq("camera_id", camera.id)
      .eq("officer_id", officerId)
      .maybeSingle();
    if (!enrol) notes.push("officer_not_enrolled_on_device");
    else if (enrol.revoked_at && new Date(enrol.revoked_at) < new Date(record.detected_at)) {
      notes.push("officer_revoked_before_capture");
    }
    if (!(await hasRole(officerId, "officer")) && !(await hasRole(officerId, "admin"))) {
      notes.push("officer_role_missing");
    }
  }

  let dutyId: string | null = record.duty_session_id ?? null;
  if (dutyId) {
    const { data: d } = await admin
      .from("duty_sessions")
      .select("id, officer_id")
      .eq("id", dutyId)
      .maybeSingle();
    if (!d) {
      notes.push("duty_session_unknown");
      dutyId = null;
    } else if (d.officer_id !== officerId) {
      notes.push("duty_session_officer_mismatch");
    }
  } else {
    notes.push("no_duty_session");
  }

  const detected = new Date(record.detected_at);
  if (isNaN(detected.getTime())) return json(400, { error: "bad_detected_at" });
  if (detected.getTime() > Date.now() + 5 * 60_000) notes.push("captured_in_future");

  for (const u of uploads) {
    const { error } = await admin.storage
      .from(BUCKET)
      .upload(u.path, u.bytes, { contentType: "image/jpeg", upsert: true });
    if (error) return json(503, { error: "storage_failed", detail: error.message });
  }

  const plate = record.plate ?? {};
  const { data: inserted, error: caseErr } = await admin
    .from("cases")
    .insert({
      case_no: caseNo,
      camera_id: camera.id,
      captured_at: detected.toISOString(),
      plate_text: plate.text || null,
      needs_manual_plate: plate.status !== "read" || !plate.text,
      status: "pending",
      recorded_by: officerId && !notes.includes("officer_role_missing") ? officerId : null,
      duty_session_id: dutyId,
      device_seq: seq,
      violation_type: record.type ?? "no_helmet",
      details: {
        violation_id: record.violation_id,
        claimed_officer_id: officerId,
        riders: record.riders ?? [],
        motorcycle: record.motorcycle ?? null,
        plate,
        models: record.models ?? null,
        app_version: record.app_version ?? null,
        location: record.location ?? null,
      },
      record_signature: signature,
      integrity_status: notes.length ? "flagged" : "verified",
      integrity_notes: notes,
    })
    .select("id")
    .single();

  if (caseErr) {
    if (caseErr.code === "23505") {
      const { data: again } = await admin
        .from("cases")
        .select("case_no")
        .eq("camera_id", camera.id)
        .eq("device_seq", seq)
        .maybeSingle();
      return json(200, { case_no: again?.case_no ?? caseNo, duplicate: true });
    }
    return json(503, { error: "case_insert_failed", detail: caseErr.message });
  }

  if (uploads.length) {
    const { error: evErr } = await admin.from("evidence").insert(
      uploads.map((u) => ({
        case_id: inserted.id,
        image_path: u.path,
        image_type: u.name,
        sha256: u.sha,
        bytes: u.bytes.length,
      })),
    );
    if (evErr) return json(503, { error: "evidence_insert_failed", detail: evErr.message });
  }

  await admin
    .from("cameras")
    .update({ last_seen_at: new Date().toISOString() })
    .eq("id", camera.id);

  return json(201, {
    case_no: caseNo,
    integrity_status: notes.length ? "flagged" : "verified",
    integrity_notes: notes,
  });
}

// ---------- entry ---------------------------------------------------------------

Deno.serve(async (req) => {
  if (req.method !== "POST") return json(405, { error: "method_not_allowed" });
  const route = new URL(req.url).pathname.split("/").filter(Boolean).pop();
  try {
    switch (route) {
      case "register":
        return await register(req);
      case "enroll":
        return await enroll(req);
      case "duty":
        return await duty(req);
      case "heartbeat":
        return await heartbeat(req);
      case "violation":
        return await violation(req);
      default:
        return json(404, { error: "unknown_route" });
    }
  } catch (e) {
    console.error("[device-api]", e);
    return json(500, { error: "internal" });
  }
});
