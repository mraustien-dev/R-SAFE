-- R-SAFE mobile detection + automatic sync
-- Adds: registered phones (on cameras), officer enrolment per phone, duty sessions,
-- tamper evidence on cases/evidence, read rules for the dashboard, realtime on cases.
-- Clients (phone app, dashboard) can NEVER insert cases/evidence directly:
-- only the `device-api` Edge Function (service role) writes them.

create extension if not exists pgcrypto with schema extensions;

-- 1. A phone = a row in cameras -----------------------------------------------
alter table public.cameras
  add column if not exists device_id            text unique,
  add column if not exists device_public_key    text,        -- base64, raw uncompressed P-256 point
  add column if not exists provision_code_hash  text,        -- sha256 hex of a one-time code
  add column if not exists provision_expires_at timestamptz,
  add column if not exists provisioned_at       timestamptz,
  add column if not exists last_seq             bigint not null default 0,  -- newest violation number the phone reports
  add column if not exists pending_count        integer not null default 0, -- violations still queued on the phone
  add column if not exists last_seen_at         timestamptz,
  add column if not exists app_version          text;

-- 2. Which officers have signed in online on which phone ------------------------
create table if not exists public.device_officers (
  camera_id   uuid not null references public.cameras(id) on delete cascade,
  officer_id  uuid not null references auth.users(id) on delete cascade,
  enrolled_at timestamptz not null default now(),
  revoked_at  timestamptz,
  primary key (camera_id, officer_id)
);

-- 3. Duty sessions (an officer on duty with a phone) -----------------------------
create table if not exists public.duty_sessions (
  id          uuid primary key,                       -- generated on the phone
  camera_id   uuid not null references public.cameras(id),
  officer_id  uuid not null references auth.users(id),
  started_at  timestamptz not null,
  ended_at    timestamptz,
  received_at timestamptz not null default now()
);
create index if not exists duty_sessions_camera_idx on public.duty_sessions (camera_id, started_at desc);

-- 4. Cases: who, which phone, which shift, integrity -----------------------------
alter table public.cases
  add column if not exists recorded_by        uuid references auth.users(id),
  add column if not exists duty_session_id    uuid references public.duty_sessions(id),
  add column if not exists device_seq         bigint,
  add column if not exists violation_type     text not null default 'no_helmet',
  add column if not exists details            jsonb,
  add column if not exists record_signature   text,
  add column if not exists received_at        timestamptz not null default now(),
  add column if not exists integrity_status   text not null default 'verified',
  add column if not exists integrity_notes    text[] not null default '{}';

do $$ begin
  alter table public.cases add constraint cases_integrity_status_chk
    check (integrity_status in ('verified', 'flagged'));
exception when duplicate_object then null; end $$;

create unique index if not exists cases_camera_seq_uidx on public.cases (camera_id, device_seq);
create index if not exists cases_captured_idx on public.cases (captured_at desc);

alter table public.evidence
  add column if not exists sha256 text,
  add column if not exists bytes  integer;

-- 5. Missing violation numbers per phone (deleted / never-sent evidence) ---------
create or replace view public.device_sequence_gaps with (security_invoker = true) as
select c.camera_id, s.seq as missing_seq
from (select camera_id, max(device_seq) as mx
      from public.cases where device_seq is not null group by camera_id) c
cross join lateral generate_series(1, c.mx) as s(seq)
where not exists (
  select 1 from public.cases x where x.camera_id = c.camera_id and x.device_seq = s.seq
);

-- 6. Admin creates a one-time code to register a phone ---------------------------
create or replace function public.create_device_provision_code(_camera_id uuid)
returns text
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
  _code text := upper(substr(encode(gen_random_bytes(6), 'hex'), 1, 8));
begin
  if not public.has_role(auth.uid(), 'admin') then
    raise exception 'admin only';
  end if;
  update public.cameras
     set provision_code_hash  = encode(digest(_code, 'sha256'), 'hex'),
         provision_expires_at = now() + interval '24 hours'
   where id = _camera_id;
  if not found then raise exception 'camera not found'; end if;
  return _code;
end $$;
revoke all on function public.create_device_provision_code(uuid) from public, anon;
grant execute on function public.create_device_provision_code(uuid) to authenticated;

-- 7. Read rules (RLS). No insert rules for clients on purpose. --------------------
-- cameras
drop policy if exists "staff read cameras" on public.cameras;
create policy "staff read cameras" on public.cameras for select to authenticated
  using (public.has_role(auth.uid(), 'admin') or public.has_role(auth.uid(), 'officer'));
drop policy if exists "admin manage cameras" on public.cameras;
create policy "admin manage cameras" on public.cameras for all to authenticated
  using (public.has_role(auth.uid(), 'admin')) with check (public.has_role(auth.uid(), 'admin'));

-- cases
drop policy if exists "admin read cases" on public.cases;
create policy "admin read cases" on public.cases for select to authenticated
  using (public.has_role(auth.uid(), 'admin'));
drop policy if exists "officer read own cases" on public.cases;
create policy "officer read own cases" on public.cases for select to authenticated
  using (recorded_by = auth.uid());
drop policy if exists "admin review cases" on public.cases;
create policy "admin review cases" on public.cases for update to authenticated
  using (public.has_role(auth.uid(), 'admin')) with check (public.has_role(auth.uid(), 'admin'));

-- evidence (visible when its case is visible)
drop policy if exists "read evidence of visible cases" on public.evidence;
create policy "read evidence of visible cases" on public.evidence for select to authenticated
  using (exists (select 1 from public.cases c where c.id = evidence.case_id));

-- duty sessions / enrolment
drop policy if exists "read duty sessions" on public.duty_sessions;
create policy "read duty sessions" on public.duty_sessions for select to authenticated
  using (public.has_role(auth.uid(), 'admin') or officer_id = auth.uid());
drop policy if exists "read device officers" on public.device_officers;
create policy "read device officers" on public.device_officers for select to authenticated
  using (public.has_role(auth.uid(), 'admin') or officer_id = auth.uid());
drop policy if exists "admin manage device officers" on public.device_officers;
create policy "admin manage device officers" on public.device_officers for update to authenticated
  using (public.has_role(auth.uid(), 'admin')) with check (public.has_role(auth.uid(), 'admin'));

alter table public.device_officers enable row level security;
alter table public.duty_sessions  enable row level security;

-- photos in the private bucket: readable by staff who can see the case
drop policy if exists "staff read case evidence" on storage.objects;
create policy "staff read case evidence" on storage.objects for select to authenticated
  using (
    bucket_id = 'case-evidence'
    and exists (select 1 from public.evidence e where e.image_path = storage.objects.name)
  );

-- 8. Live dashboard ----------------------------------------------------------------
do $$ begin
  alter publication supabase_realtime add table public.cases;
exception when duplicate_object then null; end $$;
do $$ begin
  alter publication supabase_realtime add table public.cameras;
exception when duplicate_object then null; end $$;
