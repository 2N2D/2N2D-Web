"use client";

import React, {useEffect, useState} from "react";
import {magicLink} from "@/lib/auth/authEndP";
import {useRouter} from "next/navigation";

export default function handleMail() {
    const router = useRouter()
    const [error, setError] = useState<boolean>(false);

    async function checkMail() {
        const result = await magicLink();
        if (result == "200" || result == "default") {
            router.push("/");
        } else
            setError(true);
    }

    useEffect(() => {
        checkMail();
    }, [])

    return <main>
        {
            error ? <div>
                <h1>Something went wrong</h1>
                <a href={"/login"}>Back to login</a>
            </div> : <h1>Logging in....</h1>
        }
    </main>
}