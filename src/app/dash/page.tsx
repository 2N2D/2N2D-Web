"use client"

import React, {useEffect, useState} from 'react';
import {useRouter} from "next/navigation";
import "./styles.css";
import ParticleNetwork from "@/components/visual/particleNetwork";
import {
    createSession,
    deleteSession,
    getSession,
    getUser,
    Session,
    updateUser,
    User
} from "@/lib/sessionHandling/sessionManager";
import {updateName} from "@/lib/sessionHandling/sessionUpdater";
import {motion} from "framer-motion";
import {createVisualNetwork2D, downloadFileRequest} from "@/lib/feHandler";

export default function dash() {

    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [currentSession, setCurrentSession] = useState<Session | null>(null);
    const [sessions, setSessions] = useState<Session[]>([])
    const [loading, setLoading] = useState<boolean>(false);
    const [loadingName, setLoadingName] = useState<boolean>(false);
    const [editName, setEditName] = useState<boolean>(false);
    const [downloading, setDownloading] = useState<boolean>(false);
    const [notLogged, setNotLogged] = useState<boolean>(false);

    const canvasRef = React.useRef<HTMLDivElement>(null);

    const router = useRouter();

    function updateSessionStorage(session?: Session) {
        const logged = sessionStorage.getItem("logged");
        sessionStorage.clear();
        sessionStorage.setItem("logged", logged!);

        console.log(session);

        if (session != null)
            sessionStorage.setItem("currentSessionId", session?.id.toString());

        if (session?.onnxName)
            sessionStorage.setItem("modelName", session?.onnxName);

        if (session?.csvName)
            sessionStorage.setItem("csvName", session?.csvName);

        if (session?.csvResult)
            sessionStorage.setItem("csvData", JSON.stringify(session?.csvResult));

        if (session?.visResult) {
            sessionStorage.setItem("modelResponse", JSON.stringify(session?.visResult));
            sessionStorage.setItem("modelData", JSON.stringify(session?.visResult));
        }

    }

    async function downloadOptimized() {
        setDownloading(true);
        if (!currentSession) return;

        let fileName = currentSession.onnxName!.split(".")[0] + "_optimized.onnx";
        await downloadFileRequest(currentSession.optimizedFileUrl!, "rezult", fileName)
        setDownloading(false);
    }

    async function createView() {
        if (currentSession && currentSession.visResult && (currentSession.visResult as any).nodes && JSON.stringify(currentSession?.visResult).length > 2) {
            const ctx = canvasRef.current;
            if (ctx) {
                await createVisualNetwork2D(currentSession!.visResult, ctx, false, false, false, () => {
                })
            }
        } else {
            const ctx = canvasRef.current;
            if (ctx) {
                await createVisualNetwork2D({nodes: [], edges: []}, ctx, false, false, false, () => {
                });
            }
        }
    }

    async function removeSession(id: number, index: number) {
        await deleteSession(id);

        let auxUser = currentUser;

        if (auxUser?.sessions) {
            auxUser.sessions = auxUser.sessions.filter(sessionId => sessionId !== id); // Filter out the deleted session ID
        }

        setCurrentUser(auxUser);
        await updateUser(auxUser!);


        if (sessionStorage.getItem("currentSessionId")) {
            const sesid = parseInt(sessionStorage.getItem("currentSessionId")!);
            if (id == sesid) {
                sessionStorage.removeItem("currentSessionId");
                updateSessionStorage();
            }
        }

        const auxSessions = [...sessions];
        auxSessions.splice(index, 1);
        console.log(auxSessions);
        setSessions(auxSessions);
    }

    async function CreateSession() {
        if (currentUser == null) return;

        const newSession = await createSession(currentUser);
        sessionStorage.setItem("currentSessionId", newSession.id.toString());

        setCurrentSession(newSession);
        setSessions([...sessions, newSession]);

        if (currentUser.sessions == null)
            currentUser.sessions = [];

        setCurrentUser({...currentUser, sessions: [...currentUser.sessions, newSession.id]});

        await updateUser(currentUser);
        updateSessionStorage(newSession);
    }

    async function setActiveSession(id: number) {
        setCurrentSession(null)
        const session = await getSession(id);
        if (session == null || !session) return;
        setCurrentSession(session);

        sessionStorage.setItem("currentSessionId", session.id.toString());
        updateSessionStorage(session);
    }

    async function popSessions(user: User) {
        setLoading(true);
        let sessions: Session[] = [];
        if (user.sessions != null) {
            for (const sessionId of user.sessions) {
                const session = await getSession(sessionId);
                sessions.push(session);
            }
        }
        console.log(sessions);
        setSessions(sessions);
        setLoading(false);
    }

    async function loadPage() {
        setLoadingName(true);
        setLoading(true);
        const user = await getUser();
        if (typeof user === "string") {
            setNotLogged(true);
            setLoading(false);
            setLoadingName(false);
            return;
        }
        setCurrentUser(user);
        popSessions(user);
        setLoadingName(false);
        if (sessionStorage.getItem("currentSessionId") != null) {

            setCurrentSession(await getSession(parseInt(sessionStorage.getItem("currentSessionId")!)))
        }

    }


    useEffect(() => {
        loadPage();
    }, [])

    useEffect(() => {
        createView();
    }, [currentSession]);

    useEffect(() => {
        if (currentUser == null) return;
        updateUser(currentUser)
    }, [currentUser])


    return <motion.main className={"pageDash"}
                        initial={{opacity: 0,}}
                        animate={{opacity: 1}}
                        transition={{duration: 0.6, ease: "easeOut"}}>
        <ParticleNetwork/>
        <motion.div className="lander" initial={{opacity: 0}} animate={{opacity: 1}} transition={{delay: 0.4}}>
            <div>
                <h1 className="title">Dashboard</h1>
                {notLogged ? <div>
                        <h2 className="subtitle">You are not logged in</h2>
                        <button className="notLoggedBut" onClick={() => router.push("/login")}>Log in here <i
                            className="fa-solid fa-arrow-up-right-from-square"></i></button>
                    </div> :
                    <h2 className="subtitle flex align-center gap-[0.5rem]">Hello, {loadingName ?
                        <div className="lazyLoad h-[1.5rem] w-[10rem]"/> :
                        <b>{currentUser?.displayName}!</b>}</h2>}
            </div>
            <img src="/logo2n2d.svg" alt="logo" className="logo"/>
        </motion.div>

        {
            currentSession != null ?
                <motion.div className={"currentSessionInfo"} initial={{opacity: 0, y: 20}} animate={{opacity: 1, y: 0}}
                            transition={{delay: 0.4, duration: 0.6, ease: "easeOut"}}>
                    <div>
                        {
                            editName ? <div className={"flex gap-[1rem] align-center"}>
                                <input type={"text"} defaultValue={currentSession.name ? currentSession.name : ""}
                                       onChange={async (e) => {
                                           updateName(currentSession.id, e.target.value)
                                           currentSession.name = e.target.value;
                                       }}/>
                                <button onClick={() => setEditName(false)}><i className="fa-solid fa-check"></i>
                                </button>
                            </div> : <div className={"flex gap-[1rem] align-center"}>
                                <h1 className={"title min-w-[10rem]"}>{currentSession.name}</h1>
                                <button onClick={() => setEditName(true)}><i className="fa-solid fa-pen"></i></button>
                            </div>
                        }
                    </div>
                    <div className={"element"}>
                        <h1 className={"subtitle"}>Visualize</h1>
                        {
                            <div className="networkPreview" ref={canvasRef} onClick={() => {
                                router.push("/visualize")
                            }}></div>
                        }
                        <div>
                            <h1 className={"title"}>Loaded files:</h1>
                            <h2><b>ONNX
                                file:</b> {currentSession.onnxName ? currentSession.onnxName : "No onnx file loaded"}
                            </h2>
                            <h2><b>Csv
                                file:</b> {currentSession.csvName ? currentSession.csvName : "No csv file loaded"}
                            </h2>
                        </div>
                    </div>
                    <div className={"element"}>
                        <div>
                            <h1 className={"subtitle"}> Optimization</h1>
                            {
                                currentSession && currentSession.optResult && JSON.stringify(currentSession.optResult).length > 2 && (currentSession.optResult as any).best_config ?
                                    <motion.div className={"resultArea"} initial={{opacity: 0, y: 10}}
                                                animate={{opacity: 1, y: 0}}
                                                transition={{duration: 0.4}}>
                                        <div className={"flex flex-col gap-[0.2rem] p-[1rem]"}>
                                            <h2 className={"subtitle"}>Best configuration:</h2>
                                            <div className={"result"}>
                                                <div className={"info"}>
                                                    <h2>Neurons:</h2> {(currentSession.optResult as any).best_config.neurons}
                                                </div>
                                                <div className={"info"}>
                                                    <h2>Layers:</h2> {(currentSession.optResult as any).best_config.layers}
                                                </div>
                                                <div className={"info"}><h2>Test
                                                    loss:</h2> {(currentSession.optResult as any).best_config.test_loss}
                                                </div>
                                                <div className={"info"}><h2>R2
                                                    score:</h2> {(currentSession.optResult as any).best_config.r2_score}
                                                </div>
                                            </div>
                                        </div>
                                        <button onClick={downloadOptimized}
                                                disabled={downloading}>{downloading ? "Downloading..." : "Download optimized"}
                                            <i
                                                className="fa-solid fa-file-arrow-down"></i></button>
                                    </motion.div> : <h1>No optimization done.</h1>
                            }
                        </div>
                        <div className={"flex gap-[1rem] flex-col align-center justify-center w-[20%]"}>
                            <button className={"navBut"} onClick={() => {
                                router.push("/data")
                            }}>Data <i className="fa-solid fa-chart-simple"></i></button>
                            <button className={"navBut"} onClick={() => {
                                router.push("/optimize")
                            }}>Optimization <i className="fa-solid fa-rabbit-running"></i></button>
                        </div>
                    </div>
                </motion.div> :
                <motion.div className={"flex flex-col gap-[0.1rem] p-[5rem] justify-center"} initial={{opacity: 0}}
                            animate={{opacity: 1}} transition={{delay: 0.4}}>
                    <h1 className={"title"}>No active session!</h1>
                    <h2 className={"subtitle"}>Please load or create a new session to start.</h2>
                </motion.div>
        }

        <div className={"fade"} style={{rotate: "180deg"}}/>
        <div className={"sessionsArea"}>
            <h2 className={"title"}>Sessions:</h2>
            {!loading && !notLogged ?
                <motion.div className={"sessions"} initial="hidden" whileInView="visible" variants={{
                    hidden: {opacity: 0, rotate: -2},
                    visible: {
                        opacity: 1,
                        rotate: 0,
                        transition: {
                            staggerChildren: 0.1,
                        },
                    },

                }} viewport={{once: true, amount: 0.3}}>
                    <motion.div className={"button"} onClick={CreateSession}
                                variants={{
                                    hidden: {opacity: 0, y: 20},
                                    visible: {opacity: 1, y: 0},
                                }}
                                transition={{duration: 0.2}}
                                whileHover={{
                                    scale: 1.05,
                                    rotate: 2,
                                    backgroundColor: "var(--primary-color)",
                                    color: "var(--card-background)",
                                    transition: {duration: 0.2, ease: "easeInOut"},
                                }}
                                whileTap={{
                                    scale: 0.9,
                                    rotate: -1,
                                    transition: {duration: 0.2, ease: "easeInOut"},
                                }}
                    >
                        <h2 className={"subtitle"}>Create new session</h2>
                        <i className="fa-solid fa-square-plus largeIcon"></i>
                    </motion.div>
                    {
                        sessions.length == 0 ? <div className={"empty"}></div> :
                            sessions.map((session, i) =>
                                session ?
                                    <motion.div className={"button"} key={i} onClick={() => {
                                        setActiveSession(session.id);
                                    }}
                                                variants={{
                                                    hidden: {opacity: 0, y: 20},
                                                    visible: {opacity: 1, y: 0},
                                                }}
                                                transition={{duration: 0.2}}
                                                whileHover={{
                                                    scale: 1.05,
                                                    rotate: 2,
                                                    backgroundColor: "var(--primary-color)",
                                                    color: "var(--card-background)",
                                                    transition: {duration: 0.2}
                                                }}>
                                        <div
                                            className={"flex w-full justify-between align-center "}>
                                            <div className={"flex flex-col w-[50%] overflow-hidden text-ellipsis"}>
                                                <h1 className={"subtitle"}>{session.name}</h1>
                                                <h2><b>ONNX file:</b> {session.onnxName}</h2>
                                                <h2><b>Csv file:</b> {session.csvName}</h2>
                                            </div>
                                            <i className="fa-solid fa-diagram-project largeIcon"></i>
                                        </div>
                                        <div className={"flex deleteArea justify-end align-center w-full"}>
                                            <button className={"deleteButton"} onClick={() => {
                                                removeSession(session.id, i);
                                            }}><i className={"fa-solid fa-trash-can"}/></button>
                                        </div>
                                    </motion.div> : ""
                            )
                    }
                </motion.div> : <div className="loaderSpinner"></div>}
        </div>

        <div className={"fade"}/>
    </motion.main>
}