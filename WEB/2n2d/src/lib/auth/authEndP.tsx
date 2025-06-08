"use client"
import {
    createUserWithEmailAndPassword,
    getAuth,
    GoogleAuthProvider,
    isSignInWithEmailLink,
    signInWithEmailAndPassword,
    signInWithEmailLink,
    signInWithPopup
} from "@firebase/auth";
import {initFirebaseApp} from "@/lib/firebase/firebase.config";
import {createSession} from "@/lib/auth/authentication";
import crypto from 'crypto'

export async function mailAndPass(mail: string, pass: string): Promise<string> {
    let user;
    try {
        user = await signInWithEmailAndPassword(
            getAuth(initFirebaseApp()),
            mail,
            pass
        );
        return "200";
    } catch (error) {
        console.error(error);
        return "201";
    } finally {
        if (user) {
            await createSession(await user.user.getIdToken());
        }
    }
}

export async function register(mail: string, pass: string): Promise<string> {
    try {
        await createUserWithEmailAndPassword(getAuth(initFirebaseApp()), mail, pass).then(async (userCredential) => {
            const user = userCredential.user;
            await createSession(await user.getIdToken());
            return "200"
        })
    } catch (error) {
        console.error(error);
        return "201";
    }
    return "200"
}

export async function google(): Promise<string> {
    let user;
    try {
        user = await signInWithPopup(getAuth(initFirebaseApp()), new GoogleAuthProvider());
        return "200";
    } catch (error) {
        console.error(error);
        return "201";
    } finally {
        if (user) {
            await createSession(await user.user.getIdToken());
        }
    }
}

export async function magicLink(): Promise<string> {
    const email = localStorage.getItem("email");
    if (email == null) {
        return "201 - no mail";
    }
    const auth = getAuth(initFirebaseApp());
    try {
        if (isSignInWithEmailLink(auth, window.location.href)) {
            signInWithEmailLink(auth, email, window.location.href).then(async (result) => {
                await createSession(await result.user.getIdToken());
                return "200";
            }).catch((e) => {
                console.error(e);
                return "201";
            });
        } else
            return "Not a mail window"
    } catch (error) {
        console.error(error);
        return "201"
    }
    return "default";
}

export async function getCurrentUserHash(): Promise<string> {
    const user = getAuth(initFirebaseApp()).currentUser;
    if (user != null) {
        return crypto.createHash('sha256').update(user.uid).digest('hex');
    } else
        return "0";
}