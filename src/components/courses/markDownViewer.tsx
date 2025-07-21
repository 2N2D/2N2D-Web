"use client"

import React, {useEffect, useState, useRef} from 'react';
import Style from "./markDownViewer.module.css";
import MarkdownPreview from '@uiw/react-markdown-preview';
import ScrollPercentViewer from "@/components/visual/ScrollPercentViewer";

// @ts-ignore
export default function MarkDownViewer({path, title, open, onClose}: {
    path: string,
    title: string,
    open: boolean,
    onClose: () => void
}) {
    const [md, setMd] = useState<string>("");
    const ref = useRef<HTMLDivElement>(null);
    const [loaded, setLoaded] = useState(false);

    async function loadMarkdown() {
        setLoaded(false)
        if (!path) {
            setMd("")
            return;
        }

        try {
            const res = await fetch(path);
            const text = await res.text();
            setMd(text);
            sessionStorage.setItem("screenContext", text)
        } catch (err) {
            console.error(err);
            onClose();
            setMd("");
            return;
        }

        if (ref.current) {
            ref.current.scrollTo({top: 0, left: 0, behavior: 'auto'});
        }
        setLoaded(true)
    }

    useEffect(() => {
        loadMarkdown()
    }, [path]);


    return <div className={Style.cont} style={open ? {right: "0"} : {right: "-90vw"}} ref={ref}>
        <div className={Style.bar}>
            <button onClick={() => {
                onClose()
            }}><i className="fa-solid fa-xmark"></i></button>
            <h1>{title}</h1>
            <img alt={"logo"} src={"/logo2n2dNNDEV.png"}/>
        </div>
        <MarkdownPreview source={md} style={{
            backgroundColor: "var(--card-background)",
            padding: "1rem"
        }}/>
        {
            loaded ? <ScrollPercentViewer visible={open} ref={ref} loaded={loaded}/> : ""
        }


    </div>
}
