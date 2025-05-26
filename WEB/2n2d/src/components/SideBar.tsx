"use client";

import React, {useState, useEffect} from "react";
import {usePathname} from "next/navigation";
import Styles from "./SideBar.module.css";
import {getSession, logout} from "@/lib/auth/authentication";

const SideBar = () => {
    const [open, setOpen] = React.useState(false);
    const pathname = usePathname();
    const [logged, setLogged] = React.useState<boolean>(false);

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
            {/*{*/}
            {/*    logged && pathname !== "/" && pathname !== "/singup" && pathname !== "/login" ? "" :*/}
            {/*        <div className={Styles.warning}>*/}
            {/*            <h1>You are not logged in and thus cannot use the app, please log in <a href={"/login"}>here</a>*/}
            {/*            </h1>*/}
            {/*        </div>*/}
            {/*}*/}
            <div
                className={
                    open ? Styles.container : `${Styles.container} ${Styles.closed}`
                }
                onMouseEnter={() => setOpen(true)}
                onMouseLeave={() => setOpen(false)}
            >
                <a href={"/"}>
                    <img
                        src={open ? "logo2n2d.svg" : "logo.svg"}
                        alt="logo"
                        className={Styles.logo}
                    />
                </a>
                <a
                    href="/visualize"
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
                </a>
                <a
                    href="/data"
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
                </a>
                <a
                    href="/optimize"
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
                </a>
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
                        <a href={"/login"} className={
                            pathname === "/login" || pathname === "/signup"
                                ? `${Styles.tabBut} ${Styles.active}`
                                : Styles.tabBut
                        }>
                            <span className={Styles.iconWrapper}>
                                <i className="fa-solid fa-user"></i>
                            </span>
                            <span className={`${Styles.tabText}`}>Login</span>
                        </a>
                    }
                </div>
            </div>
        </div>
    );
};

export default SideBar;
