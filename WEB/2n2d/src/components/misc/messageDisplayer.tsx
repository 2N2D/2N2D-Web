import React from "react"
import {exchange} from "@/lib/aiChat";
import styles from "./messageDisplayer.module.css";
import Message from "@/components/misc/message"

interface props {
    messages?: exchange[];
}

const MessageDisplayer = ({messages}: props) => {
    return <div className={styles.container}>
        {messages?.map((m, i) => (
            <div key={i}>
                <Message icon={"fa-solid fa-user"} sender={"You"} content={m.question}/>
                <Message icon="fa-solid fa-robot" sender={"2N2D Assistant"} content={m.answer}/>
            </div>
        ))}
    </div>
}

export default MessageDisplayer;