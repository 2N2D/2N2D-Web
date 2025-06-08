"use client"
import React, {useState, useEffect, FormEvent} from "react";
import GoogleSignInButton from "@/components/misc/GoogleSignInButton";
import OneTimeMailSignInButton from "@/components/misc/OneTimeMailSignInButton";
import {mailAndPass} from "@/lib/auth/authEndP";
import {useRouter} from "next/navigation";
import {logout} from "@/lib/auth/authentication";
import "./style.css"
import Styles from "@/components/SideBar.module.css";
import ParticleNetwork from "@/components/visual/particleNetwork";

export default function login() {
    const router = useRouter();

    const [loggedIn, setLoggedIn] = useState(false);
    const [mpAuthError, setMPAuthError] = useState(false);

    async function attemptLogin(e: FormEvent<HTMLFormElement>) {
        setMPAuthError(false);
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        const email = formData.get("email")?.toString()!;
        const password = formData.get("password")?.toString()!;

        const rez = await mailAndPass(email, password);

        if (rez === "200") {
            router.push("/dash");
        } else {
            setMPAuthError(true);
        }
    }

    useEffect(() => {
        if (sessionStorage.getItem("logged") === "true") {
            setLoggedIn(true);
        }
    }, []);

    return <main>
        <ParticleNetwork/>
        {loggedIn ? <div>
                <h1>You are already logged in, would you like to log out?</h1>
                <button onClick={() => {
                    logout()
                }}>Log out
                </button>
            </div> :
            <div className={"form"}>
                <img
                    src={"logo2n2d.svg"}
                    alt="logo"
                    className={Styles.logo}
                />
                <h1>Welcome back!</h1>
                <form onSubmit={attemptLogin}>
                    <input name={"email"} type={"email"} placeholder={"Email"} required={true}/>
                    <input name={"password"} type={"password"} placeholder={"Password"} required={true}/>
                    <input type={"submit"} value={"Login"}/>
                </form>
                <div className={mpAuthError ? "error" : "hidden"}>
                    <h1>Wrong username or password!</h1>
                </div>
                <GoogleSignInButton/>
                <OneTimeMailSignInButton/>
                <h2>Don't have an account? <a href={"/signup"}>Sign up</a></h2>
            </div>
        }

    </main>
}

