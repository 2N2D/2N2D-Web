"use client"
import {useRouter} from "next/navigation";
import {google} from "@/lib/auth/authEndP";
import Style from "./SignInButton.module.css"

const GoogleSignInButton = () => {
    const router = useRouter();

    async function tryGoogleLogin() {
        const rez = await google();
        if (rez === "200") {
            router.push("/");
        }
    }

    return <button className={Style.button} onClick={tryGoogleLogin}><i className="fa-brands fa-google"></i> Google
    </button>
}

export default GoogleSignInButton;