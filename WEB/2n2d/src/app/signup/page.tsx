"use client"
import React, {useState, useEffect, FormEvent} from "react";
import GoogleSignInButton from "@/components/misc/GoogleSignInButton";
import OneTimeMailSignInButton from "@/components/misc/OneTimeMailSignInButton";
import {registerMailAndPass} from "@/lib/auth/authEndP";
import {useRouter} from "next/navigation";
import {logout} from "@/lib/auth/authentication";
import "./style.css"
import Styles from "@/components/SideBar.module.css";

export default function signup() {
    const router = useRouter();

    const [loggedIn, setLoggedIn] = useState(false);

    async function attemptSignUp(e: FormEvent<HTMLFormElement>) {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        const email = formData.get("email")?.toString()!;
        const password = formData.get("password")?.toString()!;

        const rez = await registerMailAndPass(email, password);

        if (rez === "200") {
            router.push("/");
        }
    }

    useEffect(() => {
        if (sessionStorage.getItem("logged") === "true") {
            setLoggedIn(true);
        }
    }, []);

    return <main>
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
                <h1>Welcome!</h1>
                <form>
                    <input type={"email"} placeholder={"Email"} required={true}/>
                    <input type={"password"} placeholder={"Password"} required={true}/>
                    <input type={"submit"} value={"Sign up"}/>
                </form>
                <GoogleSignInButton/>
                <OneTimeMailSignInButton/>
                <h2>Already have an account? <a href={"/login"}>Login</a></h2>
            </div>
        }

    </main>
}

