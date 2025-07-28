'use client';
import React, { useState, useEffect, FormEvent } from 'react';
import GoogleSignInButton from '@/components/auth/GoogleSignInButton';
import OneTimeMailSignInButton from '@/components/auth/OneTimeMailSignInButton';
import { mailAndPass } from '@/lib/auth/authEndP';
import { useRouter } from 'next/navigation';
import { logout } from '@/lib/auth/authentication';
import '../style.css';
import ParticleNetwork from '@/components/visual/particleNetwork';
import Styles from '@/components/layout/SideBar.module.css';
import { useLingui, Trans } from '@lingui/react/macro';

export default function login() {
  const router = useRouter();
  const { t } = useLingui();

  const [loggedIn, setLoggedIn] = useState(false);
  const [mpAuthError, setMPAuthError] = useState(false);

  async function attemptLogin(e: FormEvent<HTMLFormElement>) {
    setMPAuthError(false);
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const email = formData.get('email')?.toString()!;
    const password = formData.get('password')?.toString()!;

    const rez = await mailAndPass(email, password);

    if (rez !== '200') {
      setMPAuthError(true);
    }
  }

  useEffect(() => {
    if (sessionStorage.getItem('logged') === 'true') {
      setLoggedIn(true);
    }
  }, []);

  return (
    <main className={'logCont'}>
      <ParticleNetwork />
      {loggedIn ? (
        <div>
          <h1>{t`You are already logged in, would you like to log out?`}</h1>
          <button
            onClick={() => {
              logout();
            }}
          >
            {t`Log out`}
          </button>
        </div>
      ) : (
        <div className={'form'}>
          <img src={'logo2n2d.svg'} alt={t`logo`} className={Styles.logo} />
          <h1>{t`Welcome back!`}</h1>
          <form onSubmit={attemptLogin}>
            <input
              name={'email'}
              type={'email'}
              placeholder={t`Email`}
              required={true}
            />
            <input
              name={'password'}
              type={'password'}
              placeholder={t`Password`}
              required={true}
            />
            <input type={'submit'} value={t`Login`} />
          </form>
          <div className={mpAuthError ? 'error' : 'hidden'}>
            <h1>{t`Wrong username or password!`}</h1>
          </div>
          <GoogleSignInButton />
          <OneTimeMailSignInButton />
          <h2>
            <Trans>
              Don't have an account? <a href={'/register'}>Register here!</a>
            </Trans>
          </h2>
        </div>
      )}
    </main>
  );
}
