'use client';
import {
  createUserWithEmailAndPassword,
  getAuth,
  GoogleAuthProvider,
  isSignInWithEmailLink,
  signInWithEmailAndPassword,
  signInWithEmailLink,
  signInWithPopup,
  UserCredential,
  sendEmailVerification
} from '@firebase/auth';
import { initFirebaseApp } from '@/lib/firebase/firebase.config';
import { createSession } from '@/lib/auth/authentication';
import {
  createUser,
  getSpecificUser
} from '@/lib/sessionHandling/sessionManager';
import { redirect } from 'next/navigation';
import { send } from 'process';

export async function mailAndPass(mail: string, pass: string): Promise<string> {
  let user;
  try {
    user = await signInWithEmailAndPassword(
      getAuth(initFirebaseApp()),
      mail,
      pass
    );
    await sendEmailVerification(user.user);
    redirect('/login');
    return '200';
  } catch (error) {
    console.error(error);
    return '201';
  } finally {
    if (user) {
      await createSession(await user.user.getIdToken());
    }
  }
}

export async function register(mail: string, pass: string): Promise<string> {
  try {
    await createUserWithEmailAndPassword(
      getAuth(initFirebaseApp()),
      mail,
      pass
    ).then(async (userCredential) => {
      const user = userCredential.user;
      await createSession(await user.getIdToken());
      await createUser(mail, mail.split('@')[0]);
      redirect('/dash');
      return '200';
    });
  } catch (error) {
    console.error(error);
    return '201';
  }
  return '200';
}

export async function google(): Promise<string> {
  let user: UserCredential;
  try {
    user = await signInWithPopup(
      getAuth(initFirebaseApp()),
      new GoogleAuthProvider()
    );

    if (user) {
      setTimeout(() => {
        redirect('/dash');
      }, 0);
    }
    return '200';
  } catch (error) {
    console.error(error);
    return '201';
  } finally {
    // @ts-ignore
    if (user) {
      await createSession(await user.user.getIdToken()).finally(async () => {
        if ((await getSpecificUser(user.user.uid)) == null)
          await createUser(user.user.email!, user.user.email!.split('@')[0]);
      });
    }
  }
}

export async function magicLink(): Promise<string> {
  const email = localStorage.getItem('email');
  if (email == null) {
    return '201 - no mail';
  }
  const auth = getAuth(initFirebaseApp());
  try {
    if (isSignInWithEmailLink(auth, window.location.href)) {
      signInWithEmailLink(auth, email, window.location.href)
        .then(async (user) => {
          await createSession(await user.user.getIdToken()).finally(
            async () => {
              if ((await getSpecificUser(user.user.uid)) == null)
                await createUser(
                  user.user.email!,
                  user.user.email!.split('@')[0]
                );
            }
          );
          redirect('/dash');
          return '200';
        })
        .catch((e) => {
          console.error(e);
          return '201';
        });
    } else return 'Not a mail window';
  } catch (error) {
    console.error(error);
    return '201';
  }
  return 'default';
}

export async function deleteAcc() {
  const auth = getAuth(initFirebaseApp());
  const user = auth.currentUser;
  if (user) {
    try {
      await user.delete();
      redirect('/login');
      return '200';
    } catch (error) {
      console.error(error);
      return '201';
    }
  } else {
    console.error('No user is currently signed in.');
    return '201';
  }
}
