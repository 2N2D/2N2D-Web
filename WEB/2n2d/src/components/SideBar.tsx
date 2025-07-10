"use client";

import React, {useState, useEffect} from "react";
import {usePathname, useRouter} from "next/navigation";
import Styles from "./SideBar.module.css";
import {getSession, logout} from "@/lib/auth/authentication";

const SideBar = () => {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();
    const [logged, setLogged] = useState<boolean>(false);
    const [sessionLoaded, setSessionLoaded] = useState<boolean>(false);
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

    function checkSession() {
        const currentSession = sessionStorage.getItem("currentSessionId");
        if (currentSession != null) {
            setSessionLoaded(true);
        } else {
            setSessionLoaded(false);
        }
    }

    useEffect(() => {
        checkLogged();
        checkSession()
    }, [pathname]);

    useEffect(() => {
        if (typeof window !== 'undefined') {
            checkSession();
        }
    }, [typeof window !== 'undefined' ? sessionStorage?.getItem("currentSessionId") : null]);

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

                <button
                    onClick={() => {
                        router.push("/dash")
                    }}
                    className={
                        pathname === "/dash"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-house"></i>
                     </span>
                    <span className={`${Styles.tabText}`}>Home</span>
                </button>

                <h2 className={Styles.tabCat}>Analyze</h2>
                <button
                    onClick={() => {
                        router.push("/visualize")
                    }}
                    className={
                        pathname === "/visualize"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                    disabled={!sessionLoaded}
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
                    disabled={!sessionLoaded}
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
                    disabled={!sessionLoaded}
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-rabbit-running"></i>
                    </span>
                    <span className={`${Styles.tabText}`}>Optimization</span>
                </button>
                <button
                    onClick={() => {
                        router.push("/modeltest")
                    }}
                    className={
                        pathname === "/modeltest"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                    disabled={!sessionLoaded}
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-chart-scatter"></i>
                    </span>
                    <span className={`${Styles.tabText}`}>Test</span>
                </button>


                <div className={Styles.spacer}/>

                <h2 className={Styles.tabCat}>Info</h2>
                <button
                    onClick={() => {
                        router.push("/learn")
                    }}
                    className={
                        pathname === "/learn"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-book-open-cover"></i>
                    </span>
                    <span className={`${Styles.tabText}`}>Learn</span>
                </button>
                <button
                    onClick={() => {
                        router.push("/docs")
                    }}
                    className={
                        pathname === "/docs"
                            ? `${Styles.tabBut} ${Styles.active}`
                            : Styles.tabBut
                    }
                >
                    <span className={Styles.iconWrapper}>
                        <i className="fa-solid fa-books"></i>
                    </span>
                    <span className={`${Styles.tabText}`}>Docs</span>
                </button>

                <div className={Styles.loginZone}>
                    {logged ?
                        <button
                            onClick={() => {
                                router.push("/profile")
                            }}
                            className={
                                pathname === "/profile"
                                    ? `${Styles.tabBut} ${Styles.active}`
                                    : Styles.tabBut
                            }
                        >
                            <span className={Styles.iconWrapper}>
                                <i className="fa-solid fa-user"></i>
                            </span>
                            <span className={`${Styles.tabText}`}>Profile</span>
                        </button> : ""
                    }

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
