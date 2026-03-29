-- MediaGuardX Migration 002: Fix RLS policy gaps and add missing indexes
-- Addresses: missing INSERT bypass for trigger, missing admin policies,
-- missing UPDATE/DELETE policies, overly permissive activity_logs INSERT,
-- missing performance indexes, and FK cascade fix.

-- ============================================
-- 1. Profiles: allow service-role / trigger inserts
-- ============================================
-- The handle_new_user() trigger runs as SECURITY DEFINER but still needs
-- an RLS policy that permits the insert. During signup auth.uid() is NULL,
-- so the existing "Users can insert own profile" policy blocks the trigger.
-- This policy allows inserts only when the id matches auth.uid() OR when
-- the caller is the service role (auth.uid() is NULL in trigger context).
create policy "Allow trigger and service-role profile inserts"
  on profiles for insert
  with check (auth.uid() = id OR auth.uid() IS NULL);

-- ============================================
-- 2. Reports: add missing admin SELECT and DELETE policies
-- ============================================
create policy "Admins and investigators can view all reports"
  on reports for select
  using (public.is_privileged());

create policy "Users can delete own reports"
  on reports for delete
  using (auth.uid() = user_id);

create policy "Admins can delete any report"
  on reports for delete
  using (public.is_admin());

-- ============================================
-- 3. Detections: add missing DELETE policies
-- ============================================
create policy "Users can delete own detections"
  on detections for delete
  using (auth.uid() = user_id);

create policy "Admins can delete any detection"
  on detections for delete
  using (public.is_admin());

-- ============================================
-- 4. Activity logs: restrict INSERT to own logs
-- ============================================
-- Replace overly permissive "Anyone can insert logs" policy.
-- The backend uses service-role key (bypasses RLS), so this only
-- restricts direct client inserts.
drop policy if exists "Anyone can insert logs" on activity_logs;
create policy "Users can insert own activity logs"
  on activity_logs for insert
  with check (auth.uid()::text = user_id::text OR user_id IS NULL);

-- ============================================
-- 5. Activity logs: fix FK cascade
-- ============================================
-- Set user_id to NULL when referenced user is deleted (preserves log history).
alter table public.activity_logs
  drop constraint if exists activity_logs_user_id_fkey;
alter table public.activity_logs
  add constraint activity_logs_user_id_fkey
  foreign key (user_id) references auth.users on delete set null;

-- ============================================
-- 6. Additional performance indexes
-- ============================================
create index if not exists idx_reports_user_id on reports(user_id);
create index if not exists idx_detections_label on detections(label);
create index if not exists idx_detections_media_type on detections(media_type);
