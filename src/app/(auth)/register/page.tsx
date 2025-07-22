'use client';
import React, { useState, useEffect, FormEvent } from 'react';
import GoogleSignInButton from '@/components/misc/GoogleSignInButton';
import OneTimeMailSignInButton from '@/components/misc/OneTimeMailSignInButton';
import { register } from '@/lib/auth/authEndP';
import { useRouter } from 'next/navigation';
import { logout } from '@/lib/auth/authentication';
import './style.css';
import Styles from '@/components/SideBar.module.css';
import ParticleNetwork from '@/components/visual/particleNetwork';
import { Trans } from '@lingui/react/macro';

export default function signup() {
  const router = useRouter();

  const [loggedIn, setLoggedIn] = useState(false);

  async function attemptSignUp(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const email = formData.get('email')?.toString();
    const password = formData.get('password')?.toString();

    if (!email || !password) {
      return;
    }

    const rez = await register(email, password);
    console.log(rez);
    if (rez === '200') {
      router.push('/profile');
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
        <div className={'signCont'}>
          <h1><Trans>You are already logged in, would you like to log out?</Trans></h1>
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
          <h1><Trans>Welcome!</Trans></h1>
          <form onSubmit={attemptSignUp}>
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
              pattern='(?=.*\d)(?=.*[\W_]).{7,}'
              title='Minimum of 7 characters. Should have at least one special character and one number.'
            />
            <input type={'submit'} value={'Sign up'} />
          </form>
          <GoogleSignInButton />
          <OneTimeMailSignInButton />
          <h2>
            <Trans>Already have an account? </Trans><a href={'/login'}><Trans>Login</Trans></a>
          </h2>
        </div>
      )}
    </main>
  );
}
