'use client';

import React, { useState, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Styles from './SideBar.module.css';
import { getSession, logout } from '@/lib/auth/authentication';
import SidebarButton from './sidebarButton';
import { Trans, useLingui } from '@lingui/react/macro';

export default function Sidebar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const [logged, setLogged] = useState<boolean>(false);
  const [sessionLoaded, setSessionLoaded] = useState<boolean>(false);
  const router = useRouter();

  const { t } = useLingui();

  async function checkLogged() {
    if ((await getSession()) == '200') {
      sessionStorage.setItem('logged', 'true');
      setLogged(true);
    } else {
      setLogged(false);
      sessionStorage.setItem('logged', 'false');
    }
  }

  function checkSession() {
    const currentSession = sessionStorage.getItem('currentSessionId');
    if (currentSession != null) {
      setSessionLoaded(true);
    } else {
      setSessionLoaded(false);
    }
  }

  useEffect(() => {
    checkLogged();
    checkSession();
  }, [pathname]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      checkSession();
    }
  }, [
    typeof window !== 'undefined'
      ? sessionStorage?.getItem('currentSessionId')
      : null
  ]);
  if (pathname === '/' || pathname.includes('/docs')) return null; // Hide sidebar on the home page
  return (
    <div>
      <div
        className={
          open ? Styles.container : `${Styles.container} ${Styles.closed}`
        }
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
      >
        <button
          onClick={() => {
            router.push('/');
          }}
        >
          <img
            src={open ? 'logo2n2d.svg' : 'logo.svg'}
            alt={t`logo`}
            className={Styles.logo}
          />
        </button>

        <SidebarButton
          icon='fa-solid fa-house'
          text={t`Dashboard`}
          active={pathname === '/dash'}
          onClick={() => router.push('/dash')}
        />

        <h2 className={Styles.tabCat}>
          <Trans>Analyze</Trans>
        </h2>
        <SidebarButton
          icon='fa-solid fa-chart-network'
          text={t`Visualize`}
          active={pathname === '/visualize'}
          disabled={!sessionLoaded}
          onClick={() => router.push('/visualize')}
        />

        <SidebarButton
          icon='fa-solid fa-chart-simple'
          text={t`Data`}
          active={pathname === '/data'}
          disabled={!sessionLoaded}
          onClick={() => router.push('/data')}
        />
        <h2 className={Styles.tabCat}>
          <Trans>Tools</Trans>
        </h2>
        <SidebarButton
          icon='fa-solid fa-rabbit-running'
          text={t`Optimization`}
          active={pathname === '/optimize'}
          disabled={!sessionLoaded}
          onClick={() => router.push('/optimize')}
        />

        <div className={Styles.spacer} />

        <h2 className={Styles.tabCat}>
          <Trans>Info</Trans>
        </h2>
        <SidebarButton
          icon='fa-solid fa-book-open-cover'
          text={t`Learn`}
          active={pathname === '/learn'}
          onClick={() => router.push('/learn')}
        />
        <SidebarButton
          icon='fa-solid fa-books'
          text={t`Docs`}
          active={pathname === '/docs'}
          onClick={() => router.push('/docs')}
        />

        <div className={Styles.loginZone}>
          {logged ? (
            <button
              onClick={() => {
                router.push('/profile');
              }}
              className={
                pathname === '/profile'
                  ? `${Styles.tabBut} ${Styles.active}`
                  : Styles.tabBut
              }
            >
              <span className={Styles.iconWrapper}>
                <i className='fa-solid fa-user'></i>
              </span>
              <span className={`${Styles.tabText}`}>
                <Trans>Profile</Trans>
              </span>
            </button>
          ) : (
            ''
          )}

          {logged ? (
            <button
              className={Styles.tabBut}
              onClick={() => {
                logout();
                checkLogged();
              }}
            >
              <span className={Styles.iconWrapper}>
                <i className='fa-solid fa-right-from-bracket'></i>
              </span>
              <span className={`${Styles.tabText}`}>
                <Trans>Logout</Trans>
              </span>
            </button>
          ) : (
            <button
              onClick={() => {
                router.push('/login');
              }}
              className={
                pathname === '/login' || pathname === '/signup'
                  ? `${Styles.tabBut} ${Styles.active}`
                  : Styles.tabBut
              }
            >
              <span className={Styles.iconWrapper}>
                <i className='fa-solid fa-user'></i>
              </span>
              <span className={`${Styles.tabText}`}>
                <Trans>Login</Trans>
              </span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
