"use client"

import React, {useState, useEffect} from 'react';
import "./styles.css";
import ParticleNetwork from "@/components/visual/particleNetwork";
import {getUser, User, createSession, getSession, Session, updateUser} from "@/lib/sessionHandling/sessionManager";

export default function dash() {

    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [currentSession, setCurrentSession] = useState<Session | null>(null);
    const [sessions, setSessions] = useState<Session[]>([])


    async function CreateSession() {
        if (currentUser == null) return;

        const newSession = await createSession(currentUser);
        sessionStorage.setItem("currentSessionId", newSession.id.toString());
        console.log(newSession);
        setCurrentSession(newSession);

        if (currentUser.sessions == null)
            currentUser.sessions = [];

        setCurrentUser({...currentUser, sessions: [...currentUser.sessions, newSession.id]});


        await updateUser(currentUser);
    }

    async function popSessions(user: User) {
        let sessions: Session[] = [];
        if (user.sessions != null) {
            for (const sessionId of user.sessions) {
                const session = await getSession(sessionId);
                sessions.push(session);
            }
        }
        setSessions(sessions);
    }

    async function loadPage() {
        const user = await getUser();
        setCurrentUser(user);

        popSessions(user);

        if (sessionStorage.getItem("currentSessionId") != null)
            setCurrentSession(await getSession(parseInt(sessionStorage.getItem("currentSessionId")!)))
    }


    useEffect(() => {
        loadPage();
    }, [])

    useEffect(() => {
        if (currentUser == null) return;
        popSessions(currentUser!);
        updateUser(currentUser)
    }, [currentUser])
    

    return <main className={"pageDash"}>
        <ParticleNetwork/>
        <div className={"lander"}>
            <div>
                <h1 className={"title"}>Dashboard</h1>
                <h2 className={"subtitle"}>Hello, <b>{currentUser?.displayName}!</b></h2>
            </div>
            <img
                src={"logo2n2d.svg"}
                alt="logo"
                className={"logo"}
            />
        </div>

        {
            currentSession != null ? <div className={"CurrentSessionInfo"}>
                <h1>Session id: {currentSession.id}</h1>
            </div> : ""
        }

        <div className={"fade"} style={{rotate: "180deg"}}/>
        <div className={"sessionsArea"}>
            <h2 className={"title"}>Sessions:</h2>
            <div className={"sessions"}>
                <button onClick={CreateSession}>
                    <h2 className={"subtitle"}>Create new session</h2>
                    <i className="fa-solid fa-square-plus"></i>
                </button>
                {
                    sessions.map((session, i) =>
                        <button key={i}>
                            <div className={"flex flex-col"}>
                                <h1 className={"subtitle"}>Session {session.id}</h1>
                                <h2><b>ONNX file:</b> {session.onnxName}</h2>
                                <h2><b>Csv file:</b> {session.csvName}</h2>
                            </div>
                            <i className="fa-solid fa-diagram-project"></i>
                        </button>
                    )
                }
            </div>
        </div>

        <div className={"fade"}>
        </div>
    </main>
}