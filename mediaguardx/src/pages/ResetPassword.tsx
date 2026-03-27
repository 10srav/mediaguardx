import { useState } from 'react';
import { Link } from 'react-router-dom';
import { supabase, isDemoMode } from '@/lib/supabase';
import AuthLayout from '@/components/layouts/AuthLayout';
import { Lock, CheckCircle, AlertCircle } from 'lucide-react';

export default function ResetPassword() {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    try {
      if (isDemoMode) {
        setSuccess(true);
        return;
      }

      // Supabase automatically picks up the recovery token from the URL hash
      // and exchanges it for a session, so we can call updateUser directly.
      const { error: updateError } = await supabase.auth.updateUser({ password });
      if (updateError) throw updateError;

      await supabase.auth.signOut();
      setSuccess(true);
    } catch (err: any) {
      setError(err.message || 'Failed to reset password. The link may have expired.');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <AuthLayout>
        <div className="card p-8 text-center">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 bg-emerald-500/15 rounded-2xl flex items-center justify-center border border-emerald-500/20">
              <CheckCircle className="w-8 h-8 text-emerald-400" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-slate-200 mb-2">Password Updated</h2>
          <p className="text-slate-500 mb-6">Your password has been successfully reset.</p>
          <Link to="/login" className="btn-primary py-3 px-6 inline-flex">
            Go to Login
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout>
      <div className="card p-8">
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 bg-indigo-500/15 rounded-2xl flex items-center justify-center border border-indigo-500/20">
            <Lock className="w-8 h-8 text-indigo-400" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-center mb-2 text-gradient-brand">New Password</h1>
        <p className="text-center text-slate-500 mb-8">Enter your new password below</p>

        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">New Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="input-field"
              placeholder="At least 8 characters"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Confirm Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              className="input-field"
              placeholder="Repeat your new password"
            />
          </div>
          <button type="submit" disabled={loading} className="w-full btn-primary py-3">
            {loading ? 'Updating...' : 'Reset Password'}
          </button>
        </form>

        <p className="mt-6 text-center text-sm">
          <Link to="/login" className="text-indigo-400 hover:text-indigo-300">
            Back to Login
          </Link>
        </p>
      </div>
    </AuthLayout>
  );
}
