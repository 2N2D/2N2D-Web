'use client';

import React, { useState, useEffect } from 'react';
import ParticleNetwork from '@/components/visual/particleNetwork';
import { useRouter } from 'next/navigation';
import {
  getUser,
  User,
  updateUser
} from '@/lib/sessionHandling/sessionManager';
import { logout } from '@/lib/auth/authentication';
import './styles.css';

export default function Profile() {
  const [user, setUser] = useState<User | null>(null);
  const router = useRouter();

  async function onLoad() {
    const sesId = sessionStorage.getItem('currentSessionId');
    if (!sesId) {
      router.push('/');
      return;
    }

    const _user = await getUser();
    if (!_user || typeof _user == 'string') {
      router.push('/');
      return;
    }

    setUser(_user);
  }

  async function updateName(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!user) return;

    const formData = new FormData(e.currentTarget);
    const updatedUser = {
      ...user,
      displayName: formData.get('username')?.toString()!
    };

    await updateUser(updatedUser);
    setUser(updatedUser);
  }

  async function handleLogout() {
    sessionStorage.removeItem('currentSessionId');
    await logout();
    router.push('/login');
  }

  useEffect(() => {
    onLoad();
  }, []);

  return (
    <main className={'profilePage'}>
      <ParticleNetwork />
      <div className={'profileArea'}>
        <h1 className={'title'}>Profile</h1>
        {user ? (
          <form onSubmit={updateName}>
            <label className={'subtitle'}>Username:</label>
            <input
              className={'subtitle'}
              type={'text'}
              defaultValue={user.displayName!}
              name={'username'}
            />
            <input type={'submit'} value={'Save'} />
          </form>
        ) : (
          <div className={'loaderSpinner'} />
        )}
        <div>
          <h2 className={'subtitle'}>E-Mail:</h2>
          <p>{user?.email}</p>
        </div>
        <button onClick={handleLogout} className={'logoutBut'}>
          Logout <i className='fa-solid fa-right-from-bracket'></i>
        </button>
      </div>
    </main>
  );
}
