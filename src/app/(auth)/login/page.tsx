'use client';
import React, { useState, useEffect, FormEvent } from 'react';
import GoogleSignInButton from '@/components/misc/GoogleSignInButton';
import OneTimeMailSignInButton from '@/components/misc/OneTimeMailSignInButton';
import { mailAndPass } from '@/lib/auth/authEndP';
import { useRouter } from 'next/navigation';
import { logout } from '@/lib/auth/authentication';
import './style.css';
import Styles from '@/components/SideBar.module.css';
import ParticleNetwork from '@/components/visual/particleNetwork';
import { Trans } from '@lingui/react/macro';

export default function login() {
  const router = useRouter();

  const [loggedIn, setLoggedIn] = useState(false);
  const [mpAuthError, setMPAuthError] = useState(false);

  async function attemptLogin(e: FormEvent<HTMLFormElement>) {
    setMPAuthError(false);
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const email = formData.get('email')?.toString()!;
    const password = formData.get('password')?.toString()!;

    const rez = await mailAndPass(email, password);

    if (rez === '200') {
      router.push('/dash');
    } else {
      setMPAuthError(true);
    }
  }

  useEffect(() => {
    if (sessionStorage.getItem('logged') === 'true') {
      setLoggedIn(true);
    }
  }, []);

  return (
    <main>
      <ParticleNetwork />
      {loggedIn ? (
        <div className={'logCont'}>
          <h1><Trans>You are already logged in, would you like to log out?</Trans></h1>
          <button
            onClick={() => {
              logout();
            }}
          >
            Log out
          </button>
        </div>
      ) : (
        <div className={'form'}>
          <img src={'logo2n2d.svg'} alt='logo' className={Styles.logo} />
          <h1><Trans>Welcome back!</Trans></h1>
          <form onSubmit={attemptLogin}>
            <input
              name={'email'}
              type={'email'}
              placeholder={'Email'}
              required={true}
            />
            <input
              name={'password'}
              type={'password'}
              placeholder={'Password'}
              required={true}
            />
            <input type={'submit'} value={'Login'} />
          </form>
          <div className={mpAuthError ? 'error' : 'hidden'}>
            <h1><Trans>Wrong username or password!</Trans></h1>
          </div>
          <GoogleSignInButton />
          <OneTimeMailSignInButton />
          <h2>
            <Trans>Don't have an account?</Trans> <a href={'/signup'}><Trans>Sign up</Trans></a>
          </h2>
        </div>
      )}
    </main>
  );
}
