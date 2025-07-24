'use client';
import React, { FormEvent, useState } from 'react';
import { getAuth, sendSignInLinkToEmail } from '@firebase/auth';
import { initFirebaseApp } from '@/lib/firebase/firebase.config';
import Styles from './SignInButton.module.css';
const OneTimeMailSignInButton = () => {
  const [state, setState] = useState<boolean>();

  const actionCodeSettings = {
    url: 'http://localhost:3000/handleMail',
    handleCodeInApp: true
  };

  async function sendMail(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const email = formData.get('email')?.toString()!;
    console.log(email);
    if (email) {
      localStorage.setItem('email', email);
      await sendSignInLinkToEmail(
        getAuth(initFirebaseApp()),
        email,
        actionCodeSettings
      );
    }
  }

  if (!state) {
    return (
      <button className={Styles.button} onClick={() => setState(true)}>
        <i className='fa-solid fa-envelope'></i> Magic Link
      </button>
    );
  } else {
    return (
      <div className={'popup'}>
        <div>
          <div className={'flex w-full justify-end'}>
            <button
              className={'ClosePopup'}
              onClick={() => {
                setState(false);
              }}
            >
              <i className='fa-solid fa-xmark-large'></i>
            </button>
          </div>
          <h1> Magic Link sign-in </h1>
          <form onSubmit={sendMail}>
            <input
              type={'email'}
              placeholder={'Email'}
              name={'email'}
              required={true}
            />
            <input type={'submit'} value={'Send'} />
          </form>
        </div>
      </div>
    );
  }
};

export default OneTimeMailSignInButton;
