'use client';

import React, { useEffect, useState } from 'react';
import { magicLink } from '@/lib/auth/authEndP';
import { useRouter } from 'next/navigation';
import { Trans } from '@lingui/react/macro';


export default function handleMail() {
  const router = useRouter();
  const [error, setError] = useState<boolean>(false);

  async function checkMail() {
    const result = await magicLink();
    if (result == '200' || result == 'default') {
      router.push('/');
    } else setError(true);
  }

  useEffect(() => {
    checkMail();
  }, []);

  return (
    <main>
      {error ? (
        <div>
          <h1><Trans>Something went wrong</Trans></h1>
          <a href={'/login'}><Trans>Back to login</Trans></a>
        </div>
      ) : (
        <h1><Trans>Logging in....</Trans></h1>
      )}
    </main>
  );
}
