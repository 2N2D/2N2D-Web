"use client";

import React, {useState, useEffect} from "react";
import {usePathname, useRouter} from "next/navigation";
import Styles from "./SideBar.module.css";
import {getSession, logout} from "@/lib/auth/authentication";

const SideBar = () => {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();
    const [logged, setLogged] = useState<boolean>(false);
    const router = useRouter();

    async function checkLogged() {
        if (await getSession() == "200") {
            sessionStorage.setItem("logged", "true");
            setLogged(true);
        } else {
            setLogged(false);
            sessionStorage.setItem("logged", "false");
        }
    }

    useEffect(() => {
        checkLogged();
    }, [pathname]);

    return (
        <div>
            <div
                className={
                    open ? Styles.container : `${Styles.container} ${Styles.closed}`
                }
                onMouseEnter={() => setOpen(true)}
                onMouseLeave={() => setOpen(false)}
            >
                <button onClick={() => {
                    router.push("/")
                }}>
                    <img
                        src={open ? "logo2n2d.svg" : "logo.svg"}
                        alt="logo"
                        className={Styles.logo}
                    />
                </button>
                <h2 className={Styles.tabCat}>Analise</h2>
                <button
                    onClick={() => {
                        router.push("/visualize")
                    }}
                    className={
                        pathname === "/visualize"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-chart-network"></i>
                     </span>
                    <span className={`${Styles.tabText}`}>Visualize</span>
                </button>

                <button
                    onClick={() => {
                        router.push("/data")
                    }}
                    className={
                        pathname === "/data"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-chart-simple"></i>
                    </span>
                    <span className={`${Styles.tabText}`}>Data</span>
                </button>
                <h2 className={Styles.tabCat}>Tools</h2>
                <button
                    onClick={() => {
                        router.push("/optimize")
                    }}
                    className={
                        pathname === "/optimize"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-rabbit-running"></i>
                    </span>
                    <span className={`${Styles.tabText}`}>Optimization</span>
                </button>

                <div className={Styles.spacer}>

                </div>
                <div className={Styles.loginZone}>
                    {logged ?
                        <button className={Styles.tabBut} onClick={() => {
                            logout();
                            checkLogged();
                        }}>
                            <span className={Styles.iconWrapper}>
                                <i className="fa-solid fa-right-from-bracket"></i>
                            </span>
                            <span className={`${Styles.tabText}`}>Logout</span>
                        </button> :
                        <button onClick={() => {
                            router.push("/login")
                        }} className={
                            pathname === "/login" || pathname === "/signup"
                                ? `${Styles.tabBut} ${Styles.active}`
                                : Styles.tabBut
                        }>
                            <span className={Styles.iconWrapper}>
                                <i className="fa-solid fa-user"></i>
                            </span>
                            <span className={`${Styles.tabText}`}>Login</span>
                        </button>
                    }
                </div>
            </div>
        </div>
    );
};

export default SideBar;
