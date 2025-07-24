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
          <h1>You are already logged in, would you like to log out?</h1>
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
          <h1>Welcome!</h1>
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
            Already have an account?
            <a href={'/login'}>Login</a>
          </h2>
        </div>
      )}
    </main>
  );
}
