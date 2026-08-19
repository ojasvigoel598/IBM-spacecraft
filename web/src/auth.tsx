import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

// Same-origin by default (/api/*): served by the Vite dev proxy locally and
// by the Vercel Python function in production. Set VITE_API_URL to point at a
// separate API origin when the backend is hosted elsewhere.
const API = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

export type AuthUser = {
  id: number
  email: string
  role: string
  email_verified: boolean
}

export type AuthStatus = 'loading' | 'anon' | 'authed'

type SignupResult = {
  message: string
  verification_required: boolean
  dev_verification_link?: string
  verification_token?: string
}

type ResetResult = {
  message: string
  dev_reset_link?: string
  reset_token?: string
}

type AuthContextValue = {
  status: AuthStatus
  user: AuthUser | null
  login: (email: string, password: string) => Promise<AuthUser>
  signup: (email: string, password: string) => Promise<SignupResult>
  verify: (token: string) => Promise<void>
  resend: (email: string) => Promise<SignupResult>
  resetRequest: (email: string) => Promise<ResetResult>
  resetConfirm: (token: string, password: string) => Promise<void>
  logout: () => Promise<void>
  apiFetch: <T = unknown>(path: string, init?: RequestInit) => Promise<T>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function readError(r: Response): Promise<string> {
  try {
    const body = await r.json()
    if (body?.detail && typeof body.detail === 'string') return body.detail
  } catch {
    /* not JSON */
  }
  return 'Something went wrong. Please try again.'
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [user, setUser] = useState<AuthUser | null>(null)

  const apiFetch = useCallback(async <T,>(path: string, init?: RequestInit): Promise<T> => {
    const r = await fetch(`${API}${path}`, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
    if (!r.ok) {
      // session expired / not authenticated -> return to the login screen
      if (r.status === 401) {
        setStatus('anon')
        setUser(null)
      }
      throw new ApiError(r.status, await readError(r))
    }
    return (await r.json()) as T
  }, [])

  const checkMe = useCallback(async () => {
    try {
      const data = await apiFetch<{ user: AuthUser }>('/api/auth/me')
      setUser(data.user)
      setStatus('authed')
    } catch {
      setUser(null)
      setStatus('anon')
    }
  }, [apiFetch])

  useEffect(() => {
    void checkMe()
  }, [checkMe])

  const login = useCallback(
    async (email: string, password: string) => {
      const data = await apiFetch<{ user: AuthUser }>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      })
      setUser(data.user)
      setStatus('authed')
      return data.user
    },
    [apiFetch],
  )

  const signup = useCallback(
    async (email: string, password: string) => {
      return apiFetch<SignupResult>('/api/auth/signup', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      })
    },
    [apiFetch],
  )

  const verify = useCallback(
    async (token: string) => {
      await apiFetch<{ verified: boolean }>('/api/auth/verify', {
        method: 'POST',
        body: JSON.stringify({ token }),
      })
    },
    [apiFetch],
  )

  const resend = useCallback(
    async (email: string) => {
      return apiFetch<SignupResult>('/api/auth/resend', {
        method: 'POST',
        body: JSON.stringify({ email }),
      })
    },
    [apiFetch],
  )

  const resetRequest = useCallback(
    async (email: string) => {
      return apiFetch<ResetResult>('/api/auth/reset', {
        method: 'POST',
        body: JSON.stringify({ email }),
      })
    },
    [apiFetch],
  )

  const resetConfirm = useCallback(
    async (token: string, password: string) => {
      await apiFetch<{ message: string }>('/api/auth/reset/confirm', {
        method: 'POST',
        body: JSON.stringify({ token, password }),
      })
    },
    [apiFetch],
  )

  const logout = useCallback(async () => {
    try {
      await apiFetch<{ message: string }>('/api/auth/logout', { method: 'POST' })
    } finally {
      setUser(null)
      setStatus('anon')
    }
  }, [apiFetch])

  const value = useMemo(
    () => ({ status, user, login, signup, verify, resend, resetRequest, resetConfirm, logout, apiFetch }),
    [status, user, login, signup, verify, resend, resetRequest, resetConfirm, logout, apiFetch],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
