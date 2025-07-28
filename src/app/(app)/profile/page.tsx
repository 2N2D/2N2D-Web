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
import { Trans, useLingui } from '@lingui/react/macro';
import LanguageSwitcher from '@/components/ui/LanguageSwitcher';

export default function Profile() {
  const [user, setUser] = useState<User | null>(null);
  const router = useRouter();
  const { t } = useLingui();

  async function onLoad() {
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
      <div className={'profileArea'}>
        <h1 className={'title'}>
          <Trans>Profile</Trans>
        </h1>
        {user ? (
          <form onSubmit={updateName}>
            <label className={'subtitle'}>
              <Trans>Username:</Trans>
            </label>
            <input
              className={'subtitle'}
              type={'text'}
              defaultValue={user.displayName!}
              name={'username'}
            />
            <input type={'submit'} value={t`Save`} />
          </form>
        ) : (
          <div className={'loaderSpinner'} />
        )}
        <div>
          <h2 className={'subtitle'}>
            <Trans>E-Mail:</Trans>
          </h2>
          <p>{user?.email}</p>
        </div>
        <div>
          <h1 className={'subtitle'}>
            <Trans>Language:</Trans>
          </h1>
          <LanguageSwitcher />
        </div>
        <button onClick={handleLogout} className={'logoutBut'}>
          <Trans>Logout</Trans>{' '}
          <i className='fa-solid fa-right-from-bracket'></i>
        </button>
      </div>
      <ParticleNetwork />
    </main>
  );
}
