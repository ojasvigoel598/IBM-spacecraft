import { useEffect, useState, type FormEvent } from 'react'
import { Satellite } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useAuth, ApiError } from '@/auth'

type Step = 'login' | 'signup' | 'reset' | 'reset2' | 'verify'

function Field({
  label,
  type = 'text',
  value,
  onChange,
  autoComplete,
  placeholder,
}: {
  label: string
  type?: string
  value: string
  onChange: (v: string) => void
  autoComplete?: string
  placeholder?: string
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs uppercase tracking-wider text-muted-foreground">{label}</span>
      <Input
        type={type}
        value={value}
        autoComplete={autoComplete}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  )
}

export default function AuthScreen() {
  const { login, signup, verify, resend, resetRequest, resetConfirm } = useAuth()

  const [step, setStep] = useState<Step>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [token, setToken] = useState('')
  const [resetToken, setResetToken] = useState<string | null>(null)
  const [newPassword, setNewPassword] = useState('')
  const [newConfirm, setNewConfirm] = useState('')
  const [devToken, setDevToken] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Email links arrive as ?vt= (verify) / ?rt= (reset) on the console root.
  // Prefill the corresponding step so the one-time token only needs a click.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const vt = params.get('vt')
    const rt = params.get('rt')
    if (vt) {
      setToken(vt)
      setStep('verify')
      setNotice('Enter the one-time verification token sent to your inbox.')
    } else if (rt) {
      setResetToken(rt)
      setStep('reset2')
      setNotice('Enter a new password — the one-time reset token is already filled in.')
    }
  }, [])

  const showError = (e: unknown, fallback: string) => {
    setError(e instanceof ApiError ? e.message : fallback)
  }

  const doLogin = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      await login(email, password)
    } catch (e) {
      showError(e, 'login failed')
    } finally {
      setBusy(false)
    }
  }

  const doSignup = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true)
    setError(null)
    setNotice(null)
    if (password !== confirm) {
      setError('passwords do not match')
      setBusy(false)
      return
    }
    try {
      const result = await signup(email, password)
      setNotice(result.message)
      if (result.verification_required) {
        setDevToken(result.verification_token ?? null)
        setStep('verify')
      }
    } catch (e) {
      showError(e, 'signup failed')
    } finally {
      setBusy(false)
    }
  }

  const doVerify = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await verify(token.trim())
      setNotice('Email verified — logging you in…')
      await login(email, password)
    } catch (e) {
      showError(e, 'verification failed')
    } finally {
      setBusy(false)
    }
  }

  const doResend = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await resend(email)
      setDevToken(r.verification_token ?? null)
      setNotice(r.message)
    } catch (e) {
      showError(e, 'resend failed')
    } finally {
      setBusy(false)
    }
  }

  const doReset = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      const r = await resetRequest(email)
      setNotice(r.message)
      if (r.reset_token) {
        setResetToken(r.reset_token)
        setStep('reset2')
      }
    } catch (e) {
      showError(e, 'reset request failed')
    } finally {
      setBusy(false)
    }
  }

  const doResetConfirm = async (ev: FormEvent) => {
    ev.preventDefault()
    setBusy(true)
    setError(null)
    if (newPassword !== newConfirm) {
      setError('passwords do not match')
      setBusy(false)
      return
    }
    try {
      await resetConfirm(resetToken ?? '', newPassword)
      setNotice('Password updated — please log in.')
      setStep('login')
      setPassword('')
    } catch (e) {
      showError(e, 'reset failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-6 flex items-center justify-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/15 ring-1 ring-primary/40">
            <Satellite className="h-5 w-5 text-primary" />
          </div>
          <div className="text-center">
            <div className="text-base font-semibold tracking-[0.2em] text-foreground">MISSIONMIND</div>
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Spacecraft Health &amp; Reliability Console
            </div>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              {step === 'login' && 'Log in'}
              {step === 'signup' && 'Create an account'}
              {step === 'verify' && 'Verify your email'}
              {step === 'reset' && 'Reset your password'}
              {step === 'reset2' && 'Choose a new password'}
            </CardTitle>
            <CardDescription className="text-[11px]">
              {step === 'login' && 'Access the mission console with your verified account.'}
              {step === 'signup' && 'Registration is protected and rate-limited; you must verify your email before accessing telemetry.'}
              {step === 'verify' && 'Enter the one-time verification token sent to your inbox.'}
              {step === 'reset' && 'Enter your email — if it is registered, a reset link is sent.'}
              {step === 'reset2' && 'Enter the one-time reset token and a new password. All sessions are revoked.'}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {step === 'login' && (
              <form onSubmit={doLogin} className="flex flex-col gap-4">
                <Field label="Email" type="email" value={email} autoComplete="email" onChange={setEmail} />
                <Field label="Password" type="password" value={password} autoComplete="current-password" onChange={setPassword} />
                {error && <p className="text-xs text-red-400">{error}</p>}
                {notice && <p className="text-xs text-emerald-400">{notice}</p>}
                <Button type="submit" disabled={busy} className="w-full">
                  {busy ? 'Signing in…' : 'Log in'}
                </Button>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <button type="button" className="hover:text-foreground" onClick={() => { setStep('signup'); setError(null) }}>
                    Create an account
                  </button>
                  <button type="button" className="hover:text-foreground" onClick={() => { setStep('reset'); setError(null) }}>
                    Forgot password?
                  </button>
                </div>
              </form>
            )}

            {step === 'signup' && (
              <form onSubmit={doSignup} className="flex flex-col gap-4">
                <Field label="Email" type="email" value={email} autoComplete="email" onChange={setEmail} />
                <Field label="Password" type="password" value={password} autoComplete="new-password" placeholder="8+ chars, letters and digits" onChange={setPassword} />
                <Field label="Confirm password" type="password" value={confirm} autoComplete="new-password" onChange={setConfirm} />
                {error && <p className="text-xs text-red-400">{error}</p>}
                <Button type="submit" disabled={busy} className="w-full">
                  {busy ? 'Creating account…' : 'Sign up'}
                </Button>
                <div className="text-center text-xs text-muted-foreground">
                  <button type="button" className="hover:text-foreground" onClick={() => { setStep('login'); setError(null) }}>
                    Already have an account? Log in
                  </button>
                </div>
              </form>
            )}

            {step === 'verify' && (
              <form onSubmit={doVerify} className="flex flex-col gap-4">
                {notice && <p className="text-xs text-emerald-400">{notice}</p>}
                {devToken && (
                  <div className="rounded-lg border border-border bg-muted/30 p-3 text-[11px] leading-relaxed text-muted-foreground">
                    <span className="font-semibold text-foreground">Development mode:</span> no SMTP is
                    configured, so your one-time verification token is shown here.
                    <div className="mt-1.5 break-all font-mono text-[10px] text-primary">{devToken}</div>
                  </div>
                )}
                <Field label="Verification token" value={token} onChange={setToken} placeholder="paste the one-time token" />
                {error && <p className="text-xs text-red-400">{error}</p>}
                <Button type="submit" disabled={busy || !token.trim()} className="w-full">
                  {busy ? 'Verifying…' : 'Verify email'}
                </Button>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <button type="button" className="hover:text-foreground" onClick={() => { void doResend(); }}>
                    Resend token
                  </button>
                  <button type="button" className="hover:text-foreground" onClick={() => setStep('login')}>
                    Back to login
                  </button>
                </div>
              </form>
            )}

            {step === 'reset' && (
              <form onSubmit={doReset} className="flex flex-col gap-4">
                <Field label="Email" type="email" value={email} autoComplete="email" onChange={setEmail} />
                {error && <p className="text-xs text-red-400">{error}</p>}
                {notice && <p className="text-xs text-emerald-400">{notice}</p>}
                <Button type="submit" disabled={busy} className="w-full">
                  {busy ? 'Sending…' : 'Send reset link'}
                </Button>
                <div className="text-center text-xs text-muted-foreground">
                  <button type="button" className="hover:text-foreground" onClick={() => { setStep('login'); setError(null) }}>
                    Back to login
                  </button>
                </div>
              </form>
            )}

            {step === 'reset2' && (
              <form onSubmit={doResetConfirm} className="flex flex-col gap-4">
                {resetToken && (
                  <div className="rounded-lg border border-border bg-muted/30 p-3 text-[11px] leading-relaxed text-muted-foreground">
                    <span className="font-semibold text-foreground">Development mode:</span> no SMTP is
                    configured, so your one-time reset token is shown here.
                    <div className="mt-1.5 break-all font-mono text-[10px] text-primary">{resetToken}</div>
                  </div>
                )}
                <Field label="Reset token" value={resetToken ?? ''} onChange={setResetToken as (v: string) => void} />
                <Field label="New password" type="password" value={newPassword} autoComplete="new-password" placeholder="8+ chars, letters and digits" onChange={setNewPassword} />
                <Field label="Confirm new password" type="password" value={newConfirm} autoComplete="new-password" onChange={setNewConfirm} />
                {error && <p className="text-xs text-red-400">{error}</p>}
                {notice && <p className="text-xs text-emerald-400">{notice}</p>}
                <Button type="submit" disabled={busy || !resetToken} className="w-full">
                  {busy ? 'Updating…' : 'Update password'}
                </Button>
                <div className="text-center text-xs text-muted-foreground">
                  <button type="button" className="hover:text-foreground" onClick={() => { setStep('login'); setError(null) }}>
                    Back to login
                  </button>
                </div>
              </form>
            )}
          </CardContent>
        </Card>

        <p className="mt-4 text-center text-[10px] text-muted-foreground">
          Session is protected by an HttpOnly cookie · requests are rate-limited · never store
          credentials in this browser
        </p>
      </div>
    </div>
  )
}
