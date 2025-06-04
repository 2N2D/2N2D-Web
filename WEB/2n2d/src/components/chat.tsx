"use client"

import React, {useState} from 'react';
import {ask, exchange} from "@/lib/aiChat";
import MessageDisplayer from "@/components/misc/messageDisplayer";
import styles from "./Chat.module.css"
import {usePathname} from "next/navigation";

export default function Chat() {
    const [question, setQuestion] = useState<string>("");
    const [messages, setMessages] = useState<exchange[]>([])
    const [open, setOpen] = useState(false);
    const pathname = usePathname();

    async function askQuestion() {
        if (!question)
            return;
        const response = await ask(question, messages, sessionStorage.getItem("modelResponse"), sessionStorage.getItem("csvData"));
        setMessages([...messages, response]);
        setQuestion("");
    }

    if (pathname == "/" || pathname == "/login" || pathname == "/signup" || pathname == "/handleMail")
        return (
            <></>
        )
    return <div className={styles.chatContainer} style={{bottom: open ? "0" : "-45vh"}}>
        <button className={styles.toggle} onClick={() => {
            setOpen(!open);
        }}>{open ? "Close" : "AI Chat"}
        </button>
        <div className={styles.chatArea}>
            <MessageDisplayer messages={messages}/>
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