'use client';
import React, { useState, useEffect, FormEvent } from 'react';
import GoogleSignInButton from '@/components/auth/GoogleSignInButton';
import OneTimeMailSignInButton from '@/components/auth/OneTimeMailSignInButton';
import { register } from '@/lib/auth/authEndP';
import { useRouter } from 'next/navigation';
import { logout } from '@/lib/auth/authentication';
import '../style.css';
import Styles from '@/components/layout/SideBar.module.css';
import ParticleNetwork from '@/components/visual/particleNetwork';
import { Trans, useLingui } from '@lingui/react/macro';

export default function signup() {
  const router = useRouter();

  const [loggedIn, setLoggedIn] = useState(false);
  const { t } = useLingui();

  async function attemptSignUp(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const email = formData.get('email')?.toString();
    const password = formData.get('password')?.toString();

    if (!email || !password) {
      return;
    }

    const rez = await register(email, password);
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
          <h1>
            <Trans>You are already logged in, would you like to log out?</Trans>
          </h1>
          <button
            onClick={() => {
              logout();
            }}
          >
            <Trans>Log out</Trans>
          </button>
        </div>
      ) : (
        <div className={'form'}>
          <img src={'logo2n2d.svg'} alt='logo' className={Styles.logo} />
          <h1>
            <Trans>Welcome!</Trans>
          </h1>
          <form onSubmit={attemptSignUp}>
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
              pattern='(?=.*\d)(?=.*[\W_]).{7,}'
              title={t`Minimum of 7 characters. Should have at least one special character and one number.`}
            />
            <input type={'submit'} value={t`Sign up`} />
          </form>
          <GoogleSignInButton />
          <OneTimeMailSignInButton />
          <h2>
            <Trans>Already have an account?</Trans>
            <a href={'/login'}>
              <Trans>Login</Trans>
            </a>
          </h2>
        </div>
      )}
    </main>
  );
}
