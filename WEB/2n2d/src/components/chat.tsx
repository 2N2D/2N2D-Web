"use client"

import React, {useRef, useState} from 'react';
import {ask, exchange} from "@/lib/aiChat";
import MessageDisplayer from "@/components/misc/messageDisplayer";
import styles from "./Chat.module.css"
import {usePathname} from "next/navigation";

export default function Chat() {
    const [question, setQuestion] = useState<string>("");
    const [messages, setMessages] = useState<exchange[]>([])
    const [status, setStatus] = useState<string>("")
    const [open, setOpen] = useState(false);
    const pathname = usePathname();

    async function askQuestion() {
        if (!question)
            return;
        setStatus("thinking...")
        const auxQuestion = question
        setQuestion("");
        let response;
        if (pathname == "/learn" || pathname == "/docs")
            response = await ask(auxQuestion, messages, null, null, sessionStorage.getItem("screenContext"));
        else
            response = await ask(auxQuestion, messages, sessionStorage.getItem("modelResponse"), sessionStorage.getItem("csvData"), null);
        setMessages([...messages, response]);
        setStatus("")

    }

    if (pathname == "/" || pathname == "/login" || pathname == "/signup" || pathname == "/handleMail" || pathname == "/dash" || pathname == "/profile")
        return (
            <></>
        )
    return <div className={styles.chatContainer} style={{bottom: open ? "0" : "-50vh"}}>
        <button className={styles.toggle} onClick={() => {
            setOpen(!open);
        }}>{open ? "Close" : "AI Chat"}
        </button>
        <div className={styles.chatArea}>
            <MessageDisplayer messages={messages}/>
            <div>
                {
                    status && status != "" ? <h1 className={styles.status}>{status}</h1> : ""
                }
            </div>
            <form onSubmit={(e) => {
                e.preventDefault();
                askQuestion()
            }}>
                <input type={"text"} placeholder={"Type here..."}
                       onChange={(e) => setQuestion(e.target.value)} value={question}/>

                <button type={"submit"}>Send <i className="fa-solid fa-paper-plane-top"></i></button>
            </form>
        </div>
    </div>
}